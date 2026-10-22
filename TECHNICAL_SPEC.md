# Technical Specification — Domain Adapter Quality & Forgetting Arena

## Research question and hypothesis

On a compact synthetic two-domain text classification task, how do full model refitting, frozen source representations and target-plus-source replay compare in target quality and source retention? Hypothesis: replay can reduce source degradation compared with target-only fitting, potentially at a target-accuracy cost.

## Problem, users and contribution

Domain shift can improve target performance while degrading source tasks. The user is an ML researcher studying transfer and retention. The contribution is a small, repeatable controlled linear-model arena; it is not an LLM adapter claim or evidence about catastrophic forgetting in neural foundation models.

## Data and prior work

Portfolio notes point to Hugging Face PEFT for parameter-efficient adaptation. To fit CPU-only 8 GB hardware, use original seeded synthetic source/target text with lexical feature drift, not pretrained weights. Dataset is generated in-memory by the project; no third-party license applies.

## Method and architecture

Vectorize text with fixed word unigram features. Compare source baseline; target-only logistic regression refit (full adaptation proxy); source-trained frozen classifier; and target+source replay refit. Evaluate target and retained source accuracy/F1, target-source delta, feature/model size and fit time over several seeds. Save exact config and per-seed results.

## Acceptance criteria and test plan

Dataset splits are disjoint and deterministic; source model metrics exist before and after adaptation; methods run and output seed-tagged JSON/CSV; tests verify splits, metrics and reproducibility. Use baseline comparison and report inconclusive outcomes without overclaiming.

## Constraints and validity

Synthetic language is intentionally small and may favor lexical cues. This is a pipeline smoke study; use licensed natural-domain corpora and actual LoRA/PEFT models for a credible domain adaptation thesis. No large download is justified on this machine.
