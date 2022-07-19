import click
import time
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np
import osm_opening_hours_humanized as hoh
from osm_opening_hours_humanized.exceptions import HOHError
from astral import LocationInfo
from astral.location import Location

from spad.common import units
from spad import stop_inference

tqdm.pandas()

pgh = Location(
    LocationInfo(
        "Pittsburgh",
        "PA",
        "America/New_York",
        40.44127718642986,
        -80.00144481122433,
    )
)


@click.command()
@click.option("--outfile", type=click.Path())
@click.option("--db", default="postgresql://")
def generate_stop_seqeunce(outfile, db):
    location = pgh  # TODO: read in from cli
    t0 = time.time()
    ssi = stop_inference.ServiceStopInference(
        v=units("8 mph"),
        d=units("200 feet"),
        t=units("5 minutes"),
        n_pings=5,
        trip_travel_distance=units("3 miles"),
        trip_travel_time=units("20 minutes"),
    )
    click.echo(f"[{time.time()-t0:0.2f}s] Getting OSM entities...")
    osm_entities, osm_open_hours = get_osm_entities(db, location)
    click.echo(f"Retrieved {len(osm_entities)} OSM entities:")
    click.echo(osm_entities.groupby("tag_name").size())
    click.echo(f"[{time.time()-t0:0.2f}s] Getting still GPS pings...")
    pings = stop_inference.get_still_pings(db)
    click.echo(f"Retrieved {len(pings)} GPS pings.")
    click.echo(f"[{time.time()-t0:0.2f}s] Inferring stops...")
    inferred_stops = get_inferred_stops(ssi, pings)
    click.echo(f"[{time.time()-t0:0.2f}s] Computing service stops...")
    service_stops = compute_service_stops(pings, inferred_stops)
    click.echo(f"Extracted {len(service_stops)} service stops.")
    click.echo(f"[{time.time()-t0:0.2f}s] Joining stops to OSM entities...")
    service_stops["stop_buffers"] = service_stops.to_crs(
        osm_entities.crs
    ).centroid.buffer(45.0)
    service_stops.set_geometry("stop_buffers", inplace=True)
    stop_entities = service_stops.sjoin(osm_entities.reset_index(), how="left")
    click.echo(f"Matched {len(stop_entities)} stop-entitiy pairs.")
    click.echo(
        f"[{time.time()-t0:0.2f}s] Computing OSM entities opening hours..."
    )
    _is_open = pd.Series(
        np.empty(len(stop_entities)), index=stop_entities.index, name="is_open"
    )
    for idx, row in iterrows(stop_entities):
        _is_open.loc[idx] = is_open(row, osm_open_hours)
    stop_entities["is_open"] = _is_open
    click.echo(f"[{time.time()-t0:0.2f}s] Saving to {outfile}...")
    stop_entities["hours"] = stop_entities.hours.apply(get_field)
    stop_entities.to_parquet(outfile)
    click.echo(f"[{time.time()-t0:0.2f}s] Done.")


def get_field(p):
    if isinstance(p, hoh.OHParser):
        return p.field
    else:
        return None


def iterrows(df: pd.DataFrame):
    n = len(df)
    for i in tqdm(range(n)):
        yield df.index[i], df.iloc[i]


def is_open(row, open_hours):
    datetime = row.localtime_start
    if isinstance(row.hours, hoh.OHParser):
        return float(row.hours.is_open(dt=datetime))
    return get_default_open_pr(row.tag, datetime, open_hours)


def get_default_open_pr(tag, datetime, open_hours, sample=10):
    oh = open_hours[open_hours.tag == tag]
    if len(oh):
        return oh.hours.apply(lambda h: h.is_open(dt=datetime)).mean()
    return 1.0


def get_osm_entities(db, loc, tags=("aeroway", "amenity", "building", "tourism")):
    osm_entities = gpd.read_postgis(
        """
            select *
            from osm_entity_tags
            where tag_name in %(tags)s
        """,
        db,
        index_col="osm_id",
        geom_col="geometry",
        params=dict(tags=tags),
    )
    osm_entities["tag"] = osm_entities.tag_name + ":" + osm_entities.tag_value
    opening_hours = pd.read_sql(
        """
        select
            osm_id,
            coalesce(oh.tag_value, st.tag_value) as opening_hours
        from (
            select * from osm_entity_tags
            where tag_name = 'opening_hours'
        ) oh
        full outer join (select * from osm_entity_tags
             where tag_name = 'service_times'
        ) st
        using (osm_id)
        """,
        db,
        index_col="osm_id",
    )
    opening_hours["hours"] = opening_hours.opening_hours.apply(
        try_parse_hours, location=loc
    )
    opening_hours.dropna(inplace=True)
    return (
        osm_entities.join(opening_hours, how="left"),
        osm_entities.join(opening_hours, how="inner"),
    )


def try_parse_hours(s, **kwds):
    try:
        return hoh.OHParser(s, **kwds)
    except HOHError:
        return None


def compute_service_stops(pings, inferred_stops):
    stops = pings.join(inferred_stops)
    stops["driver_id"] = stops["driver_id"].apply(str)
    stops["stop_count"] = (
        stops.sort_values("localized_timestamp", ascending=False)
        .groupby("driver_id")
        .apply(count_stops)
        .reset_index("driver_id", drop=True)
    )
    service_stops = gpd.GeoDataFrame(
        stops.groupby(["driver_id", "stop_count"]).agg(
            localtime_start=("localized_timestamp", "min"),
            localtime_end=("localized_timestamp", "max"),
            n_pings=("localized_timestamp", "count"),
            geometry=("geometry", lambda gs: gs.unary_union),
        ),
        geometry="geometry",
        crs=stops.geometry.crs,
    )
    service_stops["stop_hull"] = service_stops.convex_hull
    service_stops.set_geometry("stop_hull", inplace=True)
    return service_stops


def get_inferred_stops(ssi, pings):
    data = []
    for _, g in pings.groupby(["driver_id"]):
        data.append(ssi.infer_stops(g))
    return pd.concat(data)


def count_stops(stops):
    n = (stops.p_same_stop_as_next.fillna(1.0) < 0.5).cumsum()
    n[stops.p_service_stop < 0.5] = np.nan
    return n.rank(method="dense", ascending=False)


if __name__ == "__main__":
    generate_stop_seqeunce()
