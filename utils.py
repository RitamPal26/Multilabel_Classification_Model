import os
import tensorflow as tf
import numpy as np

def parse_labels(label_file_path):
    labels_dict = {}
    
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    data_lines = lines[5:] if len(lines) > 5 and "Attr" in lines[1] else lines[1:]
    
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            image_name = parts[0]
            attributes = []
            
            for attr in parts[1:5]:
                clean_attr = str(attr).strip().upper()
                if clean_attr.startswith('1'):
                    attributes.append(1.0)
                elif clean_attr.startswith('0'):
                    attributes.append(0.0)
                else:
                    attributes.append(-1.0)
                    
            labels_dict[image_name] = attributes
            
    return labels_dict

def process_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image, label

def create_dataset(image_dir, labels_dict, batch_size=32, is_training=True):
    image_paths = []
    labels = []
    
    for img_name, attrs in labels_dict.items():
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(attrs)
            
    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels, dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_masked_loss(pos_weights=None, gamma=2.0):
    def masked_focal_loss(y_true, y_pred):
        mask_value = -1.0
        
        mask = tf.cast(tf.not_equal(y_true, mask_value), tf.float32)
        y_true_safe = tf.where(tf.equal(y_true, mask_value), tf.zeros_like(y_true), y_true)
        
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_safe, logits=y_pred)
        
        pred_prob = tf.nn.sigmoid(y_pred)
        
        p_t = y_true_safe * pred_prob + (1 - y_true_safe) * (1 - pred_prob)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        if pos_weights is not None:
            weights = tf.constant(pos_weights, dtype=tf.float32)
            weight_vector = y_true_safe * weights + (1 - y_true_safe)
            bce = bce * weight_vector

        focal_loss = modulating_factor * bce
        masked_loss = focal_loss * mask 
        
        return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-5)
        
    return masked_focal_loss