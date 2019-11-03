========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        |
    * - package
      - | |commits-since|


.. |travis| image:: https://api.travis-ci.org/dHannasch/MLOps_VideoAnomalyDetection.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/dHannasch/MLOps_VideoAnomalyDetection

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/dHannasch/MLOps_VideoAnomalyDetection?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/dHannasch/MLOps_VideoAnomalyDetection

.. |commits-since| image:: https://img.shields.io/github/commits-since/dHannasch/MLOps_VideoAnomalyDetection/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/dHannasch/MLOps_VideoAnomalyDetection/compare/v0.0.0...master



.. end-badges

An example package. Generated with cookiecutter-pylibrary.

* Free software: MIT license

Installation
============

::

    pip install video-anomaly-detection

You can also install the in-development version with::

    pip install https://github.com/dHannasch/MLOps_VideoAnomalyDetection/archive/master.zip


Documentation
=============


https://www.linkedin.com/feed/update/urn:li:activity:6512538611181846528


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
