import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from utils import parse_labels, create_dataset, get_masked_loss

class BatchLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

def calculate_class_weights(labels_dict):
    """
    Calculates positive weights for the loss function to handle the skewed dataset.
    Formula: (Number of negative samples) / (Number of positive samples)
    """
    all_labels = np.array(list(labels_dict.values()))
    
    pos_counts = np.sum(all_labels == 1.0, axis=0)
    neg_counts = np.sum(all_labels == 0.0, axis=0)
    
    pos_counts = np.where(pos_counts == 0, 1, pos_counts)
    
    pos_weights = neg_counts / pos_counts
    print(f"Calculated Class Weights for Imbalance: {pos_weights}")
    return pos_weights.tolist()

def build_model():
    """Builds the EfficientNetB0 transfer learning model."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(4, activation=None)(x)

    return models.Model(inputs=base_model.input, outputs=outputs)

def main():
    image_dir = "dataset/images"
    label_file = "dataset/labels.txt"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Parsing labels and calculating dataset distributions...")
    labels_dict = parse_labels(label_file)
    pos_weights = calculate_class_weights(labels_dict)
    
    print("Building dataset pipeline...")
    train_dataset = create_dataset(image_dir, labels_dict, batch_size=32, is_training=True)

    print("Initializing EfficientNetB0 architecture...")
    model = build_model()
    
    loss_fn = get_masked_loss(pos_weights=pos_weights)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_fn)

    print("Starting training...")
    batch_loss_tracker = BatchLossCallback()
    
    model.fit(train_dataset, epochs=10, callbacks=[batch_loss_tracker])

    weights_path = os.path.join(output_dir, "model_weights.keras")
    model.save(weights_path)
    print(f"Model weights saved to {weights_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(batch_loss_tracker.batch_losses, label='Loss')
    plt.title("Aimonk_multilabel_problem")
    plt.xlabel("iteration_number")
    plt.ylabel("training_loss")
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"Loss curve plot saved to {plot_path}")
    

if __name__ == "__main__":
    main()