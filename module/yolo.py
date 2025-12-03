# import os
# import cv2
# from ultralytics import YOLO

# class YOLODetector:
#     def __init__(self, model_path="yolo.pt", output_dir="processed"):
#         """
#         Inisialisasi YOLODetector.
#         :param model_path: path ke file model YOLO
#         :param output_dir: direktori untuk menyimpan hasil deteksi
#         """
#         self.model = YOLO(model_path)
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.model.to('cpu')

#     def detect_objects(self, img_path, filename_prefix="processed"):
#         """
#         Jalankan deteksi objek dan simpan hasil ke direktori output.
#         :param img_path: path gambar input
#         :param filename_prefix: prefix nama file hasil deteksi
#         :return: path gambar hasil proses YOLO
#         """
#         # Jalankan prediksi
#         results = self.model.predict(
#             source=img_path,
#             conf=0.25,
#             device='cpu',
#             save=False,
#             verbose=False
#         )
#         result = results[0]

#         # Baca gambar
#         image = cv2.imread(img_path)

#         # Gambar bounding box di atas gambar
#         for box in result.boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Buat nama file hasil deteksi
#         base_name = os.path.basename(img_path)
#         processed_filename = f"{filename_prefix}_{base_name}"
#         processed_path = os.path.join(self.output_dir, processed_filename)

#         # Simpan hasil anotasi
#         cv2.imwrite(processed_path, image)

#         return processed_path


import os
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolo.pt", output_dir="processed"):
        """
        Inisialisasi YOLODetector.
        :param model_path: path ke file model YOLO
        :param output_dir: direktori untuk menyimpan hasil deteksi
        """
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.to('cpu')

    def detect_objects(self, img_path, filename_prefix="processed", use_heatmap=False, 
                      sigma=50, alpha=0.6):
        """
        Jalankan deteksi objek dan simpan hasil ke direktori output.
        :param img_path: path gambar input
        :param filename_prefix: prefix nama file hasil deteksi
        :param use_heatmap: jika True gunakan heatmap, jika False gunakan bounding box
        :param sigma: ukuran gaussian blur untuk heatmap (default: 50)
        :param alpha: transparansi overlay heatmap (default: 0.6)
        :return: path gambar hasil proses YOLO
        """
        # Jalankan prediksi
        results = self.model.predict(
            source=img_path,
            conf=0.25,
            device='cpu',
            save=False,
            verbose=False
        )
        result = results[0]

        # Baca gambar
        image = cv2.imread(img_path)

        if use_heatmap:
            # Gunakan heatmap
            image = self._create_heatmap(image, result, sigma, alpha)
        else:
            # Gambar bounding box di atas gambar
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Buat nama file hasil deteksi
        base_name = os.path.basename(img_path)
        mode = "heatmap" if use_heatmap else "bbox"
        processed_filename = f"{filename_prefix}_{mode}_{base_name}"
        processed_path = os.path.join(self.output_dir, processed_filename)

        # Simpan hasil anotasi
        cv2.imwrite(processed_path, image)

        return processed_path

    def _create_heatmap(self, image, result, sigma=50, alpha=0.6):
        """
        Membuat heatmap dari hasil deteksi YOLO.
        :param image: gambar input (BGR format)
        :param result: hasil deteksi dari YOLO
        :param sigma: ukuran gaussian blur
        :param alpha: transparansi overlay
        :return: gambar dengan heatmap overlay
        """
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Ekstrak bounding boxes dan confidence scores
        boxes = result.boxes
        for box in boxes:
            # Dapatkan koordinat box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Hitung center dan ukuran
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Buat mask gaussian untuk setiap deteksi
            y_coords, x_coords = np.ogrid[:h, :w]
            
            # Gaussian 2D dengan ukuran berdasarkan bounding box
            sigma_x = box_w / 4
            sigma_y = box_h / 4
            
            gaussian = np.exp(-(
                ((x_coords - cx) ** 2) / (2 * sigma_x ** 2) +
                ((y_coords - cy) ** 2) / (2 * sigma_y ** 2)
            ))
            
            # Akumulasi heatmap dengan weight berdasarkan confidence
            heatmap += gaussian * conf
        
        # Normalisasi heatmap ke range 0-1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (sigma*2+1, sigma*2+1), 0)
        
        # Convert heatmap ke colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Overlay heatmap pada gambar asli
        output = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return output


# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi detector
    detector = YOLODetector(model_path="yolo.pt", output_dir="processed")
    
    # Pilihan 1: Deteksi dengan bounding box (default)
    result_bbox = detector.detect_objects(
        img_path="path/to/image.jpg",
        filename_prefix="processed",
        use_heatmap=False
    )
    print(f"Hasil bounding box disimpan di: {result_bbox}")
    
    # Pilihan 2: Deteksi dengan heatmap
    result_heatmap = detector.detect_objects(
        img_path="path/to/image.jpg",
        filename_prefix="processed",
        use_heatmap=True,
        sigma=50,
        alpha=0.6
    )
    print(f"Hasil heatmap disimpan di: {result_heatmap}")