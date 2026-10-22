# domain-adapter-arena

Compact source/target classification experiment built for CPU-only execution, comparing source-only, target-only, replay and sparse residual methods evaluated on both source and target test sets across multiple seeds.

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) [![CI](https://github.com/Youssef-Al-eng/domain-adapter-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/Youssef-Al-eng/domain-adapter-arena/actions/workflows/ci.yml)

## Quickstart

```bash
pip install -r requirements.txt     # install (NumPy, SciPy, scikit-learn)
python scripts/run_experiment.py    # run the arena
python -m unittest discover -s tests -v    # run tests
```

## Run

Python 3.11+; NumPy, SciPy and scikit-learn are required. Tested versions are pinned in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python scripts/run_experiment.py
python -m unittest discover -s tests -v
```

Outputs: `results/metrics.json` and `results/raw.csv`. Experiment seed range is fixed to 0–4. `python src/arena.py` runs a quick three-seed demonstration.

## Interpretation

Source retention is explicitly measured after each method, so target scores alone do not imply forgetting or preservation. The tasks are generated sentence-like lexical domains and the learners are conventional logistic-regression classifiers. This is a controlled pipeline demonstration, not a foundation-model adapter comparison or claim about catastrophic forgetting. For a thesis, replace the generated text with licensed corpora and add PEFT/LoRA models; Hugging Face documents parameter-efficient methods in its [official PEFT guide](https://huggingface.co/docs/peft).

See [technical specification](TECHNICAL_SPEC.md), [final report](FINAL_REPORT.md), and [reproduction guide](docs/reproduction.md).

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff), or cite as:

```bibtex
@software{domain_adapter_arena,
  author = {Al-eng, Youssef},
  title  = {domain-adapter-arena},
  year   = {2026},
  url    = {https://github.com/Youssef-Al-eng/domain-adapter-arena}
}
```
