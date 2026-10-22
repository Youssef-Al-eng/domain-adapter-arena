# Experiment Protocol

Run five seeds with training-only vocabulary fit on source plus target training text. Compare source-only, target-only refit, source/target replay and a frozen source linear decision function plus sparse target-vocabulary residual. Evaluate every method on separate 80-example source and target sets per seed. Report mean, sample standard deviation, macro F1 and fit time. The sparse residual has only target-specific feature coefficients active; it is an interpretable linear probe, not neural LoRA. Keep all generated evaluation samples out of vocabulary and model fitting.
