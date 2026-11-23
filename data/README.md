# MaHPIC Expression Data

Eight expression workbooks are included in `mahpic_geo/`:

| GEO accession | Experiment | Host and parasite tables | Role |
| --- | --- | --- | --- |
| GSE103507 | E04R | 2 workbooks | Reference |
| GSE104101 | E24 | 2 workbooks | Reference |
| GSE104223 | E23R | 2 workbooks | Main E23 validation |
| GSE104330 | E25 | 2 workbooks | Optional external analysis |

The E04R host workbook contains DESeq2-normalized counts; the remaining seven
workbooks contain raw counts. `manifest.json` records each filename, source
record, worksheet, byte size, and SHA-256 checksum.

```bash
make verify-data
```

Please cite DeBarry JD et al., *Scientific Data* 9, 722 (2022), and the
relevant GEO series. See `NOTICE.md` for attribution and rights.
