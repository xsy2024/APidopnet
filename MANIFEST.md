# Package Contents

This package contains the code and expression tables for the MaHPIC E23
gene-panel validation. The `data/mahpic_geo/` directory holds eight GEO
workbooks; `data/manifest.json` records their checksums and sources. The
`scripts/` directory contains the experiment and table-generation programs,
`tests/` contains package checks, and `docs/` contains reproduction notes.

Run `make release-check` to verify the package and `make reproduce` to
generate the E23 tables.
