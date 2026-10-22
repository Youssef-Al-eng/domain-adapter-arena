# Final Report — Domain Adapter Quality and Forgetting Arena

## Question and method

How do simple target adaptation strategies trade target transfer against source retention? Five seeds generate sentence-like source and target text with partially shifted positive/negative lexical cues. Four TF-IDF/logistic-regression methods are compared: source-only, target-only, source-target replay and frozen-source plus sparse target residual. Each method is measured on held-out examples from both domains. The evaluation is deterministic and CPU-sized; no pretrained weights are fine-tuned.

## Results

Mean source/target accuracy: source-only .953/.518; target-only .530/.945; replay .963/.948; sparse residual .955/.938. Source-only loses target accuracy under the generated lexical shift. Target-only improves target accuracy while reducing source retention. Replay and residual balance both domains in this fixture. Five-seed standard deviations and fit times are in `results/metrics.json`; per-seed results are in `results/raw.csv`.

## Validation and limits

Two tests pass for balanced deterministic generation and both-domain/method coverage. Run `python scripts/run_experiment.py` and `python -m unittest discover -s tests -v`. The text, labels and domain shifts are generated; patterns are simple and source/target training text is used to build the shared vocabulary. This does not establish natural-language transfer or catastrophic forgetting in a foundation model.

The host is CPU-only with approximately 8.19 GB RAM. A compatible, validated generative fine-tuning checkpoint and an appropriately licensed natural source/target corpus were not available. Cached sentence encoders do not provide a completed PEFT experiment. 20 Newsgroups was considered, but the [official scikit-learn data page](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) describes acquisition without granting a corpus license; no text was copied. No LoRA result is claimed.

## Status and next step

`TESTED` for the generated linear-classifier experiment; `BLOCKED` for real pretrained/PEFT claims. Verify data terms and obtain a suitable checkpoint before fine-tuning.
