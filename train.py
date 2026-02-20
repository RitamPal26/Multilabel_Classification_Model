import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from utils import parse_labels, create_dataset, get_masked_loss

class BatchLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

def calculate_class_weights(labels_dict):
    all_labels = np.array(list(labels_dict.values()))
    pos_counts = np.sum(all_labels == 1.0, axis=0)
    neg_counts = np.sum(all_labels == 0.0, axis=0)
    pos_counts = np.where(pos_counts == 0, 1, pos_counts)
    
    pos_weights = neg_counts / pos_counts
    print(f"Calculated Class Weights for Imbalance: {pos_weights}")
    return pos_weights.tolist()

def build_model():
    inputs = layers.Input(shape=(224, 224, 3))
    
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation=None)(x) 

    model = models.Model(inputs=inputs, outputs=outputs)
    return model, base_model

def main():
    image_dir = "dataset/images"
    label_file = "dataset/labels.txt"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Parsing labels and building dataset pipeline...")
    labels_dict = parse_labels(label_file)
    pos_weights = calculate_class_weights(labels_dict)
    train_dataset = create_dataset(image_dir, labels_dict, batch_size=32, is_training=True)

    print("Initializing architecture...")
    model, base_model = build_model()
    loss_fn = get_masked_loss(pos_weights=pos_weights)
    
    batch_loss_tracker = BatchLossCallback()

    print("\n--- Starting Phase 1: Warm-up ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_fn)
    model.fit(train_dataset, epochs=5, callbacks=[batch_loss_tracker])

    print("\n--- Starting Phase 2: Fine-Tuning ---")
    base_model.trainable = True
    
    for layer in base_model.layers[:-27]:
        layer.trainable = False
        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss_fn)

    early_stopper = EarlyStopping(
        monitor='loss', 
        patience=5, 
        min_delta=0.001, 
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        train_dataset, 
        epochs=50, 
        callbacks=[batch_loss_tracker, early_stopper]
    )

    weights_path = os.path.join(output_dir, "model_weights.keras")
    model.save(weights_path)
    print(f"\nModel weights safely saved to {weights_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(batch_loss_tracker.batch_losses, label='Loss')
    phase_1_iterations = 5 * len(train_dataset)
    plt.axvline(x=phase_1_iterations, color='r', linestyle='--', label='Phase 2 Start')
    
    plt.title("Aimonk_multilabel_problem")
    plt.xlabel("iteration_number")
    plt.ylabel("training_loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"Loss curve plot saved to {plot_path}")

if __name__ == "__main__":
    main()