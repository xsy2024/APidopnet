# Reproducibility Notes

## Data and Preprocessing

The main analysis reads the GSE104223 E23R host and parasite raw-count
workbooks in `data/mahpic_geo/`. The E25 files are available for the optional
external-data workflow. E04R and E24 are included as reference datasets;
the E04R host workbook contains DESeq2-normalized counts.

Sample-label numeric timepoints are aggregated by animal and interpolated to
the default grid `[1, 2, 3, 4, 5, 7]`. The harmonic basis uses
`OMEGA = 2*pi/48` on that grid.

## Experimental Design

The gene panel is selected once using all E23 animals, including the animal
subsequently held out in each evaluation fold. Feature selection is therefore
shared across folds. For each fold, scalers, APidopnetwork regeneration, and
FiNN are fitted on the remaining animals. Results evaluate this specified
protocol; feature selection is not an independent training-fold step.

The observed-window parameter determines the available target history for the
held-out animal. Training animals supply complete trajectories. FiNN and the
conditional-input baseline use observed conditioning expression at each
prediction timepoint; the autoregressive baseline uses previous target
expression and time features.

## Evaluation and Outputs

The default run uses seeds 2024, 2025, and 2026. MSE and R-squared are
computed for each seed and animal fold. Summary means and standard deviations
pool these seed-by-fold scores. With a single future timepoint in a held-out
animal, the R-squared denominator is zero; the corresponding CSV entry is NaN.
The tables use MSE.

Run:

```bash
make reproduce
```

The final tables are:

```text
results/mahpic_e23_gene_panel_repro/final/tables/input_ablation_table.tex
results/mahpic_e23_gene_panel_repro/final/tables/input_ablation_summary.csv
results/mahpic_e23_gene_panel_repro/final/tables/augmentation_table.tex
results/mahpic_e23_gene_panel_repro/final/tables/augmentation_summary.csv
```

The input ablation covers FiNN, autoregressive, and conditional-input models
with RNN, LSTM, and GRU cells. The augmentation comparison covers no
augmentation, bootstrap, harmonic regeneration, and APidopnetwork
regeneration under FiNN-RNN.

Run `make release-check` for package and data checks, or `make data-smoke` for
a short execution of the real-data pipeline with reduced settings.
