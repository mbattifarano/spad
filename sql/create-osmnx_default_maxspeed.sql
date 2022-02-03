create table osmnx_default_maxspeed as
select pm.fed_aid_ur as fed_urban_level, l.highway, avg(maxspeed) as maxspeed, count(*) as n_links
from osmnx_links l
left join pa_municipalities pm on st_intersects(st_centroid(l.geometry), st_transform(pm.geom, st_srid(l.geometry)))
where l.maxspeed is not NULL
group by 1, 2;
