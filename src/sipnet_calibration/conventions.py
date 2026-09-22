"""Metadata conventions every processed product in this project follows.

One place for the constants that have to agree across products, so that two
ingests cannot declare different ones. Each product's module imports what it
needs from here and re-exports it, so a caller reading about the constraints
or the initial conditions still finds the constant beside that product.

Contents
--------
:data:`CF_CONVENTIONS`
    The value of the ``Conventions`` attribute each product's netCDF carries.
"""

from __future__ import annotations

__all__ = ["CF_CONVENTIONS"]

#: The Climate and Forecast conventions the processed netCDFs declare, written
#: to the ``Conventions`` attribute and checked on load. CF governs the
#: coordinate and attribute vocabulary the products use: ``units``,
#: ``long_name``, ``standard_name`` on ``lon``/``lat``, and no ``_FillValue``
#: on a coordinate.
CF_CONVENTIONS = "CF-1.11"
