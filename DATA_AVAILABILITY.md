# Data Availability

The package includes eight MaHPIC expression workbooks from GEO series
GSE103507, GSE104101, GSE104223, and GSE104330, stored in
`data/mahpic_geo/`. The main E23 validation uses the GSE104223 host and
parasite raw-count workbooks; the E25 files support the optional external-data
workflow. The E04R host workbook contains DESeq2-normalized counts.

Original filenames, source links, file sizes, and SHA-256 checksums are listed
in `data/manifest.json`. Run `make verify-data` to check the bundled files.

Please cite DeBarry JD et al., *Scientific Data* 9, 722 (2022), and the
relevant GEO accession. Data attribution and rights are listed in
`data/NOTICE.md`.
