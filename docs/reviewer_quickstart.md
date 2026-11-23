# Reviewer Quickstart

## Install and Check

```bash
python -m pip install -r requirements.txt
make release-check
make data-smoke
```

The package includes the E23 host and parasite workbooks required by the main
analysis. `make release-check` verifies file integrity, imports, and the toy
workflow. `make data-smoke` exercises real-data model and table-generation
paths with reduced settings.

## Generate Tables

```bash
make reproduce
```

The resulting tables are in
`results/mahpic_e23_gene_panel_repro/final/tables/`. The default command runs
the animal-level E23 evaluation with three seeds. See `reproducibility.md`
for the feature-selection and prediction protocols.
