# FiNN-APidopnetwork MaHPIC E23 Reproducibility Package

Code and data for the MaHPIC E23 gene-panel validation of FiNN and
APidopnetwork. The repository includes eight GEO expression workbooks covering
E04R, E23R, E24, and E25. The main E23 analysis uses the two GSE104223
workbooks. File provenance and checksums are in `data/manifest.json`.

Repository: https://github.com/xsy2024/APidopnet

## Quickstart

Install dependencies and check the package:

```bash
python -m pip install -r requirements.txt
make release-check
```

Run a short experiment with the included data:

```bash
make data-smoke
```

Generate the E23 validation tables with the default settings:

```bash
make reproduce
```

Outputs are written to `results/mahpic_e23_gene_panel_repro/final/tables/`:

```text
input_ablation_table.tex
input_ablation_summary.csv
augmentation_table.tex
augmentation_summary.csv
```

For a synthetic example, run `make toy`. For conda installation, use
`environment.yml`. `requirements-tested.txt` lists the direct dependency
versions used for the package checks.

## Analysis Design

The E23 analysis evaluates one held-out animal at a time. A common gene panel
is selected from the full E23 cohort before splitting; scaling, regeneration,
and model fitting use the training animals in each fold. FiNN combines previous
target expression with observed contemporaneous conditioning expression to
predict the next target value. The input ablation compares FiNN,
autoregressive, and conditional-input variants under RNN, LSTM, and GRU cells.
The augmentation analysis compares observed training data, bootstrap,
harmonic, and APidopnetwork regeneration.

The default run uses seeds 2024, 2025, and 2026. Tables report mean and
standard deviation over seed-by-animal-fold scores. See
`docs/reproducibility.md` for data processing and metric definitions.

## Repository Contents

| Path | Contents |
| --- | --- |
| `data/mahpic_geo/` | Eight MaHPIC expression workbooks |
| `data/manifest.json` | File provenance and SHA-256 checksums |
| `scripts/` | E23 experiment and table-generation programs |
| `examples/` | Synthetic example |
| `tests/` | Data and workflow checks |
| `docs/` | Reproduction notes and reviewer quickstart |

The source data are described by DeBarry JD et al., *Scientific Data* 9, 722
(2022), and the relevant GEO series. Software usage is governed by `LICENSE`;
data attribution and rights are described in `data/NOTICE.md`.
