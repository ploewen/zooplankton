# Zooplankton Hierarchical Classification

## Overview
This repository contains a hierarchical image-classification pipeline for zooplankton and related particle classes in Great Lakes microscopy imagery.

The core approach combines:
- a CNN encoder (ResNet18) for visual feature extraction
- an RNN decoder (LSTM) for taxonomy-path prediction

Instead of predicting only a flat class label, the model predicts a sequence through a taxonomy (for example Root -> Copepoda -> Calanoid). This supports evaluation at both leaf level (fine class) and parent level (major group).

## Dataset
The dataset is organized under Zooplankton-Data and includes:
- Classified image folders by sampling event and replicate
- CSV summaries for sample metadata
- Processed image subsets used for modeling

The active modeling workflow in this repository uses the following leaf classes:
- Bosminidae
- Daphnia
- Rotifer
- Nauplius_Copepod
- Cyclopoid
- Harpacticoid
- Calanoid
- Bubbles
- Exoskeleton
- Fiber_Hairlike
- Fiber_Squiggly
- Plant_Matter

## Repository Layout
```text
.
├── model/
│   ├── model.py                # HierarchicalCNNRNN model definition and taxonomy constants
│   └── cnn_rnn_weights.pt      # Saved model weights
├── notebooks/
│   ├── helpers.py              # Dataset utilities and shared helpers
│   ├── pre_process_data.ipynb  # Data cleaning and preparation workflow
│   ├── eda.ipynb               # Exploratory data analysis
│   └── tuned_model.ipynb       # Training, hyperparameter tuning, and evaluation
├── Zooplankton-Data/           # Raw, CSV, and processed dataset assets
├── pyproject.toml              # Project metadata and dependencies
└── README.md
```

## Environment Setup
This project is configured with pyproject.toml.

1. Create and activate a Python environment (Python 3.13+).
2. Install dependencies with your preferred pyproject-compatible tool.
3. Open the repository in VS Code and select the environment as the notebook kernel.

Example using uv:
```bash
uv sync
```

## Recommended Workflow
1. Run notebooks/pre_process_data.ipynb to prepare modeling inputs.
2. Run notebooks/eda.ipynb to inspect class distributions and image characteristics.
3. Run notebooks/tuned_model.ipynb to:
	- define taxonomy and hierarchy tools
	- build train/validation/test splits
	- tune hyperparameters
	- train final model
	- evaluate with classification reports and confusion matrices

## Model Summary
The model in model/model.py is a hierarchical CNN-RNN:
- CNN encoder: pre-trained ResNet18 with classifier head removed.
- Projection layer: maps encoder features to decoder hidden state.
- Decoder: LSTM predicts hierarchy tokens autoregressively.
- Loss: token-level cross-entropy with padding tokens ignored.

## Notes
- Weighted sampling is used to reduce class imbalance during training.
- Evaluation includes both leaf-level and parent-level confusion matrices.
- Parent-level performance can remain strong even when fine-grained copepod subtypes are confused.
