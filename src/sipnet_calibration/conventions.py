"""Metadata conventions shared by the project's processed products.

Contents
--------
:data:`CF_CONVENTIONS`
    The ``Conventions`` attribute the netCDF products declare.

Notes
-----
A constant lives here once it has to agree across products, so that two
ingests cannot declare different values of it. A product's own module imports
what it needs and re-exports it, so a caller reading about the constraints or
the initial conditions still finds the constant beside that product. The
netCDF products -- the constraints and the initial conditions -- are the ones
this currently covers; the site table is a CSV and declares nothing.
"""

from __future__ import annotations

__all__ = ["CF_CONVENTIONS"]

#: The Climate and Forecast conventions the processed netCDFs declare, written
#: to the ``Conventions`` attribute and checked on load. CF governs the
#: coordinate and attribute vocabulary the products use: ``units``,
#: ``long_name``, ``standard_name`` on ``lon``/``lat``, and no ``_FillValue``
#: on a coordinate.
CF_CONVENTIONS = "CF-1.11"
