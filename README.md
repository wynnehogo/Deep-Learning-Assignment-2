# Unified-OneHead Multi-Task Challenge
Author: 何淑雯 RE6137012

This repository contains the implementation of a single-head multi-task model for object detection, semantic segmentation, and image classification, submitted as part of Assignment 2 for the Deep Learning course.

## Contents

- `best_model.ipynb`: Main notebook with data loaders, model definition, training, and evaluation.
- `report.md`: Formal report with architecture, schedule, results, and analysis.
- `scripts/`: Placeholder directory for data preparation and evaluation.
- `llm_dialogs.zip`: Log of ChatGPT generated discussions.
- `README.md`: Reproduction instructions.

## Setup Instructions

1. Prepare the data by running the code in 'scripts/' and combine it into one .zip file
2. Open the notebook in Google Colab.
3. Run all cells sequentially to:
   - Import datasets
   - Define and train the model
   - Evaluate performance on all three tasks

The notebook is designed to run end-to-end within 2 hours.

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- `timm`
- `torchvision`
- `pycocotools.coco`
- `numpy`, `PIL`, `scikit-learn`

All dependencies are either preinstalled in Colab or installed via the notebook.

## Notes

- Evaluation metrics (mIoU, mAP@0.3, Top-1 accuracy) are computed inside the notebook.
- All training and validation datasets are prepared using the code provided in scripts/ (including dataset references)
- Model architecture respects all constraints: <8M parameters, ≤150 ms inference, ≤2 h training.

