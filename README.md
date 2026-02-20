# Aimonk Multilabel Classification Project

This repository contains a deep learning pipeline for a multilabel image classification task and trains a model to detect the presence of 4 independent attributes.

## Project Structure

```text
Multilabel_Classification_Model/
│
├── dataset/                    # The images folder is not uploaded, you will have to make it
│   └── labels.txt              # The provided annotation file
│
├── train.py                    # Main script: Builds the dataset, trains the model, saves outputs
├── inference.py                # Predicts attributes for a single input image
├── utils.py                    # Core engine: Custom loss functions, data parsers, and loaders
│
├── outputs/                    # Auto-generated during training
│   ├── model_weights.keras     # The saved deep learning model
│   └── loss_curve.png          # Training loss vs. iteration number
│
├── README.md                   # Setup and execution guide
└── APPROACH.md                 # Detailed explanation of architecture, data handling, and loss mathematics

```

## Setup Instructions

1. **Create a Virtual Environment (Recommended):**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```


2. **Install Dependencies:**
```bash
python3 -m pip install -r requirements.txt  # On Windows: pip install -r requirements.txt
```


3. **Ensure Data is Placed Correctly:** Verify that your images are inside `dataset/images/` and the text file is exactly at `dataset/labels.txt`.

## Execution Commands

### To Run Inference

To test the model on a single image and print the detected attributes, use the inference script.

```bash
python inference.py dataset/images/image_0.jpg
```

*Note: Because the model uses Focal Loss and heavy regularization, the confidence scores are highly calibrated. You can adjust the detection sensitivity using the optional threshold flag (default is 0.5):*

```bash
python inference.py dataset/images/image_63.jpg --threshold 0.45
```

*For a detailed breakdown see [APPROACH.md](https://github.com/RitamPal26/Multilabel_Classification_Images/blob/main/APPROACH.md).*
