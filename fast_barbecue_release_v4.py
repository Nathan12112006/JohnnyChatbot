#!/usr/bin/env python3
"""Bitmap-only combined Commons collector for the barbecue release."""

import fast_barbecue_release as base
import fast_barbecue_release_v3 as combined

TERMS = [
    "people actively grilling barbecue",
    "person cooking on barbecue grill",
    "people grilling meat",
    "person grilling",
    "people grilling",
    "barbecue cooking",
    "barbecue cook",
    "barbecue festival",
    "barbecue competition",
    "pitmaster barbecue",
    "outdoor grill cooking",
    "backyard barbecue",
    "park barbecue",
    "beach barbecue",
    "riverside barbecue",
    "lakeside barbecue",
    "tailgate grilling",
    "cookout grill",
    "barbecue",
    "barbeque",
    "barbecuing",
    "barbequing",
    "BBQ grilling",
    "grill cook",
    "braai",
    "churrasco",
    "asado parrilla",
    "mangal barbecue",
    "kebab grilling",
    "satay grilling",
    "yakitori grilling",
]

combined.QUERIES[:] = [
    f"{term} filetype:bitmap filew:>320 fileh:>320" for term in TERMS
]

if __name__ == "__main__":
    raise SystemExit(base.main())
