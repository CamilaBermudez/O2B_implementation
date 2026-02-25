# Online Learning to Batch Learning Experiment

This project implements Online Gradient Descent (OGD) for logistic regression and demonstrates the Online-to-Batch conversion theorem. It compares the performance of online learning algorithms with batch Empirical Risk Minimization (ERM).

## Overview

The notebook implements:
- **Data Generation**: i.i.d. synthetic data for binary classification
- **Online Learning**: Online Gradient Descent with learning rate decay
- **Online-to-Batch Conversion**: Uniform averaging of iterates
- **Evaluation**: Risk curves, generalization bounds, and cumulative regret

## Key Results

The experiment demonstrates:
1. **Training Risk Convergence**: How online-to-batch averaging converges
2. **Test Risk (Generalization)**: Comparison of generalization performance
3. **Cumulative Regret**: Online algorithm achieving O(√T) regret bound

## Requirements

- numpy
- matplotlib
- scikit-learn
- scipy

## Usage

Run the Jupyter notebook to reproduce all results.

## References

This implementation is based on online learning theory and the online-to-batch conversion theorem from machine learning literature.
