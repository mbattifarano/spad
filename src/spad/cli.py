import click
from pathlib import Path
from sqlalchemy.engine import create_engine
import pint
import psycopg2 as pg
import datetime as dt
import logging
from . import db
from .common import SPADError
from .bounding_box import BoundingBox
from .map_matching import create_map_match_tables, map_match_trajectories


units = pint.UnitRegistry()


logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s %(levelname)s %(name)s@%(lineno)d %(funcName)s: %(message)s"
)

log = logging.getLogger(__name__)


@click.group(context_settings={'show_default': True})
@click.option('-v', '--verbose', count=True, help="Set verbosity")
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


@spad.command()
@click.option("--max-ping-time-delta", type=TimeInterval(), required=True)
@click.option("--min-trajectory-duration", type=TimeInterval(), required=True)
@click.option("--max-match-distance", type=float, required=True, help="(meters)")
@click.option("--db-uri", default="postgresql://", help="Database URI.")
@click.option("--subnetwork-buffer", type=float, default=2000.0, help="(meters)")
@click.option("--lazy-load-network", is_flag=True, default=False)
@click.option("--transition-exp-scale", type=float, default=1.0)
@click.option("--limit", type=int, default=None)
@click.option("--commit-every", type=int, default=None)
def map_match(max_ping_time_delta, min_trajectory_duration, max_match_distance, db_uri, subnetwork_buffer, lazy_load_network, transition_exp_scale, limit, commit_every):
    with pg.connect(db_uri) as conn:
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
        )

class CLIError(SPADError):
    pass
