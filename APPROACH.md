# Architectural Strategy & Problem Solving

This document outlines the engineering decisions made to satisfy the specific constraints of the Aimonk multilabel dataset, specifically addressing missing data ("NA") and severe class imbalance.

## 1. Architectural Strategy
The core model utilizes **Transfer Learning** via the `EfficientNet-B0` architecture. EfficientNet-B0 was chosen because it offers an exceptionally high accuracy-to-parameter ratio, making it highly resistant to overfitting on imbalanced datasets compared to heavier models like ResNet-50.

* **Base:** Pre-trained ImageNet weights (top classification layer removed).
* **Head:** A `GlobalAveragePooling2D` layer, followed by a `Dropout(0.3)` layer for regularization, and a final `Dense` layer with 4 output neurons (no activation function applied here for mathematical stability in the loss function).
* **Two-Phase Fine-Tuning:** 1. **Warm-up:** Trained only the custom head for 5 epochs to prevent random initial weights from destroying the pre-trained feature maps.
  2. **Fine-Tuning:** Unfroze the top 27 layers of the EfficientNet base, reduced the learning rate from 1e-3 to 1e-4, and trained using an Early Stopping callback to deeply adapt the network to the specific attributes of this dataset.

## 2. Handling Missing Annotations ("NA" Values)
The dataset contained rows tagged with "NA", indicating missing ground-truth information for specific attributes. Deleting these images would discard valuable data, and replacing "NA" with `0` would introduce false negatives.

**Solution: Custom Masked Loss Function**
During data parsing, "NA" strings were converted to a mathematical masking value (`-1.0`). Inside the custom TensorFlow loss calculation, a boolean mask dynamically zeros out the loss for any attribute labeled `-1.0`. This mathematically forces the network to only update its weights based on known data (0s and 1s), safely ignoring the missing gaps while still learning from the other present attributes in that same image.

## 3. Handling the Skewed / Imbalanced Dataset
Multilabel datasets often suffer from extreme class imbalances, causing models to heavily bias toward common attributes. I tackled this through two integrated techniques:

* **Dynamic Class Weighting:** The data loader pipeline dynamically calculates the ratio of negative to positive samples for each of the 4 attributes before training begins. Rare attributes receive a proportionally massive weight multiplier.
* **Masked Focal Loss:** To prevent the massive class weights from destabilizing the training loop (causing massive error spikes), the custom masked loss function was upgraded to a **Masked Focal Loss**. This applies a modulating factor to the cross-entropy calculation. It down-weights the error for attributes the model confidently recognizes, forcing the optimizer to focus its learning capacity strictly on the hard, minority classes.

## 4. Pre-processing & Augmentations
To prevent overfitting and force the model to learn true visual concepts rather than memorizing exact pixel layouts, **Data Augmentation** was integrated directly into the TensorFlow graph via Keras preprocessing layers:
* `RandomFlip("horizontal_and_vertical")`
* `RandomRotation(0.2)`

## 5. Future Improvements (Unimplemented due to time constraints)
Given more time, I would implement the following techniques to further boost accuracy:
* **Optimal Threshold Calibration:** Instead of relying on manual probability thresholds during inference, I would use a validation set to plot a Precision-Recall curve and mathematically calculate the optimal, independent probability threshold for *each* of the 4 attributes.
* **Advanced Augmentations (Cutout):** Implementing Cutout (randomly masking square regions of the input image) forces the network to look at the entire image context rather than relying on a single distinguishing feature for a specific attribute.
* **MLSMOTE:** For extreme class imbalances, applying Multi-Label Synthetic Minority Over-sampling Technique (MLSMOTE) to synthetically generate new feature combinations for the rarest attributes before training.
