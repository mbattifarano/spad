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


