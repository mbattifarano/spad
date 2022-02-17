Simultaneous Preference and Demand Learning for Transportation Networks (SPAD)
==============================================================================

.. image:: https://travis-ci.com/mbattifarano/spad.svg?token=vzx77hPQNs9ah2CCneKd&branch=main
    :target: https://travis-ci.com/mbattifarano/spad

A python package for learning heterogeneous traveller preferences and travel
demand in transportation networks from data.

Development Setup
=================

This project uses poetry_ for python packaging and dependency management.

1. `Install <https://python-poetry.org/docs/#installation>`_ the poetry cli.
2. Clone this repository.
3. From the top level of this project run, ``poetry install``.
4. Verify the setup by running the test suite: ``poetry run pytest``.


To build the documentation:

1. ``cd docs``
2. ``make html``

To generate or update the API documentation run, ``sphinx-apidoc -o docs/source src/``

.. _poetry: https://python-poetry.org/


Database Setup
==============

The GPS processing and map-matching code requires an installation of postgres with
the postgis extension enabled.

To load the osmnx road network data into the database use `spad import-osmnx-network`

To load the GPS traces into the database use `spad import-gps`

Raw open street maps data is used to extract nearby places. First download the 
relevant osm xml data file (data.osm) then import it with:

`osm2pgsql -c data.osm -p swpa --hstore --output-pgsql-schema osm`

Additionally, the PA municipalities shapefile will need to be loaded into the database.
The data is availabe from the
`Pennsylvania spatial data access webpage <https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=41>`_
Once downloaded and decompressed the sql script `sql/import-pa-muni.sql` can be used to 
load the data into a table.

Finally, default speeds for each link in the network are computed and stored in the
`osmnx_default_maxspeed` table which can be created by running the
`sql/create-osmnx_default_maxspeed.sql` script.
