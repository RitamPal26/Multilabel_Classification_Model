import numpy as np
import tensorflow as tf
from utils import parse_labels, create_dataset
from sklearn.metrics import precision_recall_fscore_support

def main():
    model_path = "outputs/model_weights.keras"
    image_dir = "dataset/images"
    label_file = "dataset/labels.txt"

    print("Loading model and dataset...")
    model = tf.keras.models.load_model(model_path, compile=False)
    labels_dict = parse_labels(label_file)
    
    eval_dataset = create_dataset(image_dir, labels_dict, batch_size=1, is_training=False)

    y_true_all = []
    y_pred_all = []

    print("Running evaluation (ignoring NA values)...")
    for images, labels in eval_dataset:
        logits = model.predict(images, verbose=0)
        probs = tf.nn.sigmoid(logits).numpy()[0]
        true_labels = labels.numpy()[0]

        preds = (probs >= 0.45).astype(int)

        y_true_all.append(true_labels)
        y_pred_all.append(preds)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    print("\n--- Per-Attribute Performance ---")
    for i in range(4):
        mask = y_true_all[:, i] != -1.0
        y_true_masked = y_true_all[mask, i]
        y_pred_masked = y_pred_all[mask, i]

        if len(y_true_masked) == 0:
            print(f"Attr{i+1}: No valid labels found for evaluation.")
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_masked, y_pred_masked, average='binary', zero_division=0
        )

        print(f"Attr{i+1}: Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")

if __name__ == "__main__":
    main()