PYTHON ?= python
SEEDS ?= 8

.PHONY: help setup verify quick simulation support robustness real-data stats figures clean

help:
	@echo "Targets: setup verify quick simulation support robustness real-data stats figures clean"
	@echo "Override defaults with: make simulation SEEDS=8 PYTHON=python3"

setup:
	bash setup.sh

verify:
	$(PYTHON) scripts/verify_project.py

quick: verify
	$(PYTHON) -m genai_dt.experiments --quick --seeds 1 --out results/quick.json

simulation:
	$(PYTHON) -m genai_dt.experiments --seeds $(SEEDS) --out results/results.json

support:
	$(PYTHON) -m genai_dt.support_analysis --seeds $(SEEDS)

robustness:
	$(PYTHON) -m genai_dt.robustness --seeds $(SEEDS)

real-data:
	$(PYTHON) -m realdata.real_experiments --seeds $(SEEDS)

stats:
	$(PYTHON) analysis/analyse.py

figures:
	$(PYTHON) analysis/figures.py
	$(PYTHON) analysis/real_figure.py

clean:
	rm -f results/quick.json results/robustness.json results/support.json results/real_results.json
	rm -rf build dist *.egg-info
