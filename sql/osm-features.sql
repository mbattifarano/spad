create table osmnx_link_entities as 
select l.u, l.v, l.key, e.id
from osmnx_links l
join osm.swpa_line e on st_dwithin(l.geometry, st_transform(e.way, 32617), 100.0)
union all (
select l.u, l.v, l.key, e.id
from osmnx_links l
join osm.swpa_point e on st_dwithin(l.geometry, st_transform(e.way, 32617), 100.0)
)
union all (
select l.u, l.v, l.key, e.id
from osmnx_links l
join osm.swpa_polygon e on st_dwithin(l.geometry, st_transform(e.way, 32617), 100.0)
)
union all (
select l.u, l.v, l.key, e.id
from osmnx_links l
join osm.swpa_roads e on st_dwithin(l.geometry, st_transform(e.way, 32617), 100.0)
);