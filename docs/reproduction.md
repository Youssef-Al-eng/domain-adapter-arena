# Reproduction

Python 3.11+ with the versions in `requirements.txt`. From this directory run `python -m pip install -r requirements.txt`, `python scripts/run_experiment.py`, and `python -m unittest discover -s tests -v`. The main result is `results/metrics.json`; per-seed measurements are in `results/raw.csv`. Five seeds are encoded in the runner.
