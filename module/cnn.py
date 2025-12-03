import tensorflow as tf
import numpy as np
import os
import json

with open('cnn/class_indices.json', 'r') as f:
    class_indices = json.load(f)

CLASS_NAMES = list(class_indices.keys())

class CNNDetector:
    def __init__(self, model_path):
        
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            self.model = None


    def detect_objects(self, img_path):
        """
        Jalankan deteksi objek dan simpan hasil ke direktori output.
        :param img_path: path gambar input
        :param class_names: daftar nama kelas untuk klasifikasi
        :return: path gambar hasil proses YOLO
        
        """
        # Implementasi deteksi objek menggunakan model CNN
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = self.model.predict(img_array)
        # predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_class]
        confidence = np.max(predictions)
        
        return predicted_class, confidence