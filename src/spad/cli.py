import click
from pathlib import Path
from sqlalchemy.engine import create_engine
import pint
import psycopg2 as pg
import datetime as dt
import logging
import pandas as pd
import time

from tqdm import tqdm
from . import db
from .common import SPADError
from .bounding_box import BoundingBox
from .map_matching import create_map_match_tables, map_match_trajectories
from .stop_inference import ServiceStopInference, get_still_pings

units = pint.UnitRegistry()


logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s %(levelname)s %(name)s@%(lineno)d %(funcName)s: %(message)s",  # noqa
)

log = logging.getLogger(__name__)


@click.group(context_settings={"show_default": True})
@click.option("-v", "--verbose", count=True, help="Set verbosity")
def spad(verbose):
    """SPAD command line interface."""
    _set_verbose(verbose)


def _set_verbose(verbose):
    levels = [
        logging.WARN,
        logging.INFO,
        logging.DEBUG,
    ]
    try:
        level = levels[verbose]
    except IndexError:
        level = logging.DEBUG
    logging.getLogger().setLevel(level)
    log.info(f"Set log level to {level}")


@spad.command()
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    default=False,
    help="Recursively search for csv files.",
)
@click.option("--db-uri", default="postgresql://", help="Database URI.")
@click.argument("infiles", type=click.Path(exists=True))
def import_gps(recursive, db_uri, infiles):
    """Import gridwise GPS traces to a postgis database."""
    path = Path(infiles)
    fnames = []
    if path.is_file():
        log.info(f"Importing CSV file {path}.")
        if not path.name.endswith("csv"):
            raise CLIError(f"Input file must be a CSV, got: {path}")
        fnames = [path]
    elif path.is_dir():
        log.info(
            f"Importing CSV files {'recursively ' if recursive else ''}"
            f"from {path}"
        )
        glob = "*.csv"
        if recursive:
            glob = "**/" + glob
        fnames = list(path.glob(glob))
    else:
        raise CLIError(f"Invalid input: {path}")
    con = create_engine(db_uri)
    db.import_trajectories(con, fnames)


@spad.command()
@click.option(
    "--north",
    type=float,
    required=True,
    help="Northern boundary of area of interest",
)
@click.option(
    "--south",
    type=float,
    required=True,
    help="Southern boundary of area of interest",
)
@click.option(
    "--east",
    type=float,
    required=True,
    help="Eastern boundary of area of interest",
)
@click.option(
    "--west",
    type=float,
    required=True,
    help="Western boundary of area of interest",
)
@click.option("--db-uri", default="postgresql://", help="Database URI.")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Over-write tables if they exist.",
)
def import_osmnx_network(north, south, east, west, db_uri, force):
    """Import an OSMnx road network into a postgis database."""
    bbox = BoundingBox(west, south, east, north)
    con = create_engine(db_uri)
    db.import_osmnx(con, bbox, force)


class TimeInterval(click.ParamType):
    name = "Time Interval"

    def convert(self, value, param, ctx):
        try:
            seconds = units(value).to("seconds").magnitude
        except pint.errors.PintError:
            self.fail(f"{value!r} must be pint parsable (ex: '4 minutes')")
        return dt.timedelta(seconds=seconds)


class QuantityType(click.ParamType):
    name = "Quantity"

    def convert(self, value, param, ctx):
        try:
            return units(value)
        except pint.errors.PintError:
            self.fail(f"{value!r} must be pint parsable (ex: '4 minutes')")
            raise


@spad.command()
@click.option(
    "--max-ping-time-delta",
    type=TimeInterval(),
    required=True,
    help="Longest time interval between two pings on the same segment.",
)
@click.option(
    "--min-trajectory-duration",
    type=TimeInterval(),
    required=True,
    help="Minimum duration of a segment.",
)
@click.option(
    "--max-match-distance",
    type=float,
    required=True,
    help=(
        "The maximum distance in meters between a gps ping and its matched "
        "link."
    ),
)
@click.option("--db-uri", default="postgresql://", help="Database URI.")
@click.option(
    "--subnetwork-buffer",
    type=float,
    default=2000.0,
    help=(
        "The buffer in meters around the trajectory to use as the local road "
        "network."
    ),
)
@click.option(
    "--cache-path",
    default=None,
    type=click.Path(),
    help=(
        "LevelDB cache directory; if not specified a directory named after "
        "the run uuid is used."
    ),
)
@click.option(
    "--start-with-driver",
    default=None,
    help="Skip all driver UUIDs lexographically preceeding this one.",
)
@click.option(
    "--lazy-load-network",
    is_flag=True,
    default=False,
    help="Load a smaller network for each gps trajectory segment.",
)
@click.option(
    "--transition-exp-scale",
    type=float,
    default=1.0,
    help=(
        "The scale parameter of the exponential distribution which defines "
        "the HMM transition probability between candidate links of successive "
        "GPS pings."
    ),
)
@click.option(
    "--accuracy-scale",
    type=float,
    default=1.0,
    help=(
        "Scale the accuracy by this factor. Larger values will increase the "
        "probability of matching a ping to further links."
    ),
)
@click.option(
    "--limit", type=int, default=None, help="Number of segments to map-match."
)
@click.option(
    "--commit-every", type=int, default=None, help="Commit interval."
)
@click.option(
    "--network-gpickle",
    type=str,
    default=None,
    help="networkx road network cache.",
)
def map_match(
    max_ping_time_delta,
    min_trajectory_duration,
    max_match_distance,
    db_uri,
    subnetwork_buffer,
    cache_path,
    start_with_driver,
    lazy_load_network,
    accuracy_scale,
    transition_exp_scale,
    limit,
    commit_every,
    network_gpickle,
):
    conn = pg.connect(db_uri)
    try:
        create_map_match_tables(conn)
        map_match_trajectories(
            conn,
            max_ping_time_delta=max_ping_time_delta,
            min_trajectory_duration=min_trajectory_duration,
            max_match_distance=max_match_distance,
            transition_exp_scale=transition_exp_scale,
            subnetwork_buffer=subnetwork_buffer,
            limit=limit,
            commit_every=commit_every,
            lazy_load_network=lazy_load_network,
            start_at_driver=start_with_driver,
            cache_path=cache_path,
            accuracy_scale=accuracy_scale,
            network_gpickle_file=network_gpickle,
        )
    except:  # noqa: E722 do not use bare 'except'
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


@spad.command()
@click.option("--db-uri", default="postgresql://", help="Database URI.")
@click.option(
    "--mean-stop-speed",
    type=QuantityType(),
    default="5 mph",
    help="Mean speed measurement when the vehicle is stopped.",
)
@click.option(
    "--stop-radius",
    type=QuantityType(),
    default="10 meters",
    help="Mean distance between points at a service stop.",
)
@click.option(
    "--stop-timedelta",
    type=QuantityType(),
    default="1 minute",
    help="Mean time interval between consecutive stop pings.",
)
@click.option(
    "--pings-per-stop",
    type=int,
    default=5,
    help="Mean number of pings associated with a stop.",
)
@click.option(
    "--trip-travel-distance",
    type=QuantityType(),
    default="3 miles",
    help="Mean trip travel distance.",
)
@click.option(
    "--trip-travel-time",
    type=QuantityType(),
    default="15 minutes",
    help="Mean trip travel time.",
)
@click.option("--outfile")
def infer_stops(
    db_uri: str,
    mean_stop_speed: pint.Quantity,
    stop_radius: pint.Quantity,
    stop_timedelta: pint.Quantity,
    pings_per_stop: int,
    trip_travel_distance: pint.Quantity,
    trip_travel_time: pint.Quantity,
    outfile: str,
):
    """Cluster still GPS pings into stops."""
    ssi = ServiceStopInference(
        v=mean_stop_speed,
        d=stop_radius,
        t=stop_timedelta,
        n_pings=pings_per_stop,
        trip_travel_distance=trip_travel_distance,
        trip_travel_time=trip_travel_time,
    )
    click.echo(f"Querying still GPS pings...")
    t0 = time.time()
    pings = get_still_pings(db_uri)
    click.echo(f"Retrieved still GPS pings in {time.time()-t0} seconds.")
    data = []
    for _, g in tqdm(pings.groupby(["driver_id"])):
        data.append(ssi.infer_stops(g))
    inferred_stops = pd.concat(data)
    if outfile:
        fname = f"{outfile}.parquet"
        click.echo(f"Saving results to {fname}.")
        inferred_stops.to_parquet(fname)
    click.echo("done")


class CLIError(SPADError):
    pass
