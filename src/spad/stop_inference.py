"""Service Stop Inference"""
from audioop import avg
from logging import warning
import numpy as np
from scipy.stats import halfnorm, lognorm, expon, logistic
import pandas as pd
import geopandas as gpd
import pint
from .common import units

ACTIVITY_CONFIDENCE = "activity_confidence"
SPEED = "speed"
DX = "dx_meters"
DT = "dt_seconds"


def get_still_pings(db_uri: str):
    return gpd.read_postgis(SQL, db_uri, geom_col="geometry", index_col="id")


class ServiceStopInference:
    def __init__(
        self,
        v: pint.Quantity,
        d: pint.Quantity,
        t: pint.Quantity,
        trip_travel_distance: pint.Quantity,
        trip_travel_time: pint.Quantity,
        n_pings: int,
        avg_speed: float = 14.48,
        std_speed: float = 8.85,
    ):
        self.v = v.to("meters per second")
        self.d = d.to("meters")
        self.t = t.to("seconds")
        self.trip_distance = trip_travel_distance.to("meters")
        self.trip_time = trip_travel_time.to("seconds")
        self.n = n_pings
        self.avg_speed = avg_speed
        self.std_speed = std_speed

    def infer_stops(self, pings: pd.DataFrame):
        pings = pings.sort_values(by="localized_timestamp")
        stops = pd.DataFrame(
            {"p_service_stop": np.exp(self._logp_service_stop(pings))},
            index=pings.index,
        )
        stops["p_same_stop_as_next"] = pd.Series(
            self._p_same_stop_marginal(pings), index=pings.index[:-1]
        )
        return stops

    def speed_dist(self):
        return to_lognorm(self.avg_speed, self.std_speed)

    def _logp_service_stop(self, pings: pd.DataFrame):
        p_stop = activity_probability(pings)
        v = pings[SPEED].values
        _p_speed_stop = halfnorm(scale=np.pi * self.v.m / np.sqrt(2)).logpdf(
            v
        ) + np.log(p_stop)
        _p_speed_not_stop = self.speed_dist().logpdf(v) + np.log(1 - p_stop)
        # return _p_speed_stop - np.logaddexp(_p_speed_stop, _p_speed_not_stop)
        return logistic(loc=self.v.m).logsf(v)

    def _logp_same_stop(self, pings: pd.DataFrame):
        t = pings[DT].values[:-1]
        d = pings[DX].values[:-1]
        _p_stop_time = expon(scale=1 / self.t.m).logpdf(t)
        _p_trip_time = logistic(loc=self.t.m).logcdf(t)
        _p_stop_distance = expon(scale=1 / self.d.m).logpdf(d)
        _p_trip_distance = logistic(loc=self.d.m).logcdf(d)

        _num = (
            _p_stop_time
            + _p_stop_distance
            + np.log(self.n)
            - np.log(self.n + 1)
        )
        _denom = np.logaddexp(
            _num, 1 - np.log(self.n + 1)
        )
        x = t * self.avg_speed + d
        return logistic(loc=self.t.m * self.avg_speed + self.d.m).logsf(x)

    def _p_same_stop_marginal(self, pings: pd.DataFrame):
        _p_stop = self._logp_service_stop(pings)
        _p_both_stops = _p_stop[:-1] + _p_stop[1:]
        return np.exp(self._logp_same_stop(pings))# + _p_both_stops)


def to_lognorm(mean: float, stddev: float):
    exp_mu = mean ** 2 / np.sqrt(mean ** 2 + stddev ** 2)
    sigma = np.log(1 + mean ** 2 / stddev ** 2)
    return lognorm(scale=exp_mu, s=sigma)


def activity_probability(pings: pd.DataFrame):
    eps = 1e-8
    return (pings[ACTIVITY_CONFIDENCE].values / 100.0).clip(eps, 1 - eps)


def nan_warn(a: np.ndarray, name: str):
    if np.isnan(a).any():
        warning(f"Found nan values in {name}: {a}")


SQL = """
select
    id,
    driver_id,
    shift_id,
    localized_timestamp,
    case when speed < 0 then
        avg(speed) filter (where speed >= 0) over ()
    else
        speed
    end as speed,
    activity_confidence,
    extract(epoch from
        (lead(localized_timestamp, 1) over (w) - localized_timestamp)
    ) as dt_seconds,
    st_distance(
        geometry::geography,
        lead(geometry::geography, 1) over w
    ) as dx_meters,
    geometry
from gps_still
where shift_id >= 0 and localized_timestamp >= '2020-01-01'
window w as (partition by driver_id order by localized_timestamp)
"""
