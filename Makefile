PYTHON ?= python
DATA_DIR ?= data/mahpic_geo
OUT_DIR ?= results/mahpic_e23_gene_panel_repro
MPLCONFIGDIR ?= /tmp/finn_apidopnetwork_mplconfig
PYTHONDONTWRITEBYTECODE ?= 1
export MPLCONFIGDIR
export PYTHONDONTWRITEBYTECODE

.PHONY: help check toy test verify-data data-smoke dry-run reproduce tables-only release-check

help:
	@echo "Targets:"
	@echo "  check       - verify that required Python packages import"
	@echo "  toy         - run the synthetic toy example"
	@echo "  test        - run automated smoke tests"
	@echo "  verify-data - verify bundled data checksums and workbook integrity"
	@echo "  data-smoke  - run reduced real-data checks (not manuscript results)"
	@echo "  dry-run     - print full reproduction commands without running them"
	@echo "  reproduce   - run full MaHPIC E23 reproduction"
	@echo "  tables-only - assemble tables from an existing work directory"
	@echo "  release-check - run package hygiene checks and smoke tests"

check:
	$(PYTHON) -c "import numpy, pandas, sklearn, scipy, matplotlib, openpyxl, torch; print('Environment OK')"

toy:
	$(PYTHON) examples/run_toy_gene_panel.py --out-dir results/toy_example

test:
	$(PYTHON) tests/test_smoke.py
	$(PYTHON) -m unittest tests.test_data_integrity

verify-data:
	$(PYTHON) scripts/verify_data.py --data-dir "$(DATA_DIR)"

data-smoke:
	$(PYTHON) tests/test_real_data.py --data-dir "$(DATA_DIR)"

dry-run:
	$(PYTHON) scripts/reproduce_mahpic_e23_gene_panel.py --data-dir $(DATA_DIR) --out-dir $(OUT_DIR) --dry-run

reproduce:
	$(PYTHON) scripts/reproduce_mahpic_e23_gene_panel.py --data-dir $(DATA_DIR) --out-dir $(OUT_DIR)

tables-only:
	$(PYTHON) scripts/reproduce_mahpic_e23_gene_panel.py --tables-only --out-dir $(OUT_DIR)

release-check:
	$(PYTHON) scripts/check_release.py
	$(MAKE) check PYTHON=$(PYTHON)
	$(MAKE) test PYTHON=$(PYTHON)
	$(MAKE) dry-run PYTHON=$(PYTHON) DATA_DIR=$(DATA_DIR) OUT_DIR=$(OUT_DIR)
	$(PYTHON) scripts/check_release.py
