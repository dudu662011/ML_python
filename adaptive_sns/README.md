# Adaptive Squash-Norm-Sigmoid Normalization

This example demonstrates a simple implementation of an adaptive
**Squash-Norm-Sigmoid (SNS)** normalization method with tunable
parameters. The code is self contained and does not depend on external
machine learning libraries.

## Contents

- `adaptive_sns.py` – Python implementation including:
  - generation of dummy multimodal data (image and tabular features)
  - the `AdaptiveSquashNormSigmoid` class
  - a tiny logistic regression model
  - a random-search routine that mimics hyperparameter optimization
  - an end-to-end demo

## Usage

Run the demo directly:

```bash
python3 adaptive_sns.py
```

The script prints the best parameters found on a validation split and the
resulting test accuracy. It serves as a toy illustration for the method
outlined in the accompanying explanation.
