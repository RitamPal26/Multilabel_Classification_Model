import argparse
import os
import tensorflow as tf
import numpy as np

def load_and_preprocess_image(image_path):
    "Reads image from disk, decodes it, resizes it to 224x224 pixels, and formats it into a tensor."
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    
    image = tf.expand_dims(image, 0)
    return image

def main():
    parser = argparse.ArgumentParser(description="Predict attributes for a single image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--model_path", type=str, default="outputs/model_weights.keras", help="Path to the trained model weights.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for detecting an attribute.")
    args = parser.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    print(f"Processing image: {args.image_path}")
    image_tensor = load_and_preprocess_image(args.image_path)

    logits = model.predict(image_tensor, verbose=0)
    
    probabilities = tf.nn.sigmoid(logits).numpy()[0]

    present_attributes = []
    for i, prob in enumerate(probabilities):
        if prob >= args.threshold:
            present_attributes.append(f"Attr{i+1}")

    print("\n--- Results ---")
    if present_attributes:
        print(f"Attributes present: {', '.join(present_attributes)}")
    else:
        print("No attributes detected above the confidence threshold.")
        
    print(f"(Raw Probabilities: {np.round(probabilities, 3)})")

if __name__ == "__main__":
    main()