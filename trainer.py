# trainer.py
import cv2
import numpy as np
from PIL import Image
import os

def train_model():
    dataset_path = 'dataset/'
    model_path = 'models/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("\n[INFO] Mempersiapkan data untuk training...")

    face_samples = []
    ids = []
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]

    for image_path in image_paths:
        pil_img = Image.open(image_path).convert('L') 
        img_numpy = np.array(pil_img, 'uint8')
        person_id = int(os.path.split(image_path)[-1].split(".")[1])
        face_samples.append(img_numpy)
        ids.append(person_id)

    if not face_samples:
        print("[ERROR] Dataset kosong. Tidak ada yang bisa dilatih.")
        return

    print(f"\n[INFO] Melatih model... Mohon tunggu.")
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(f'{model_path}face-model.yml')

    print(f"\n[INFO] Model berhasil dilatih dan disimpan sebagai 'face-model.yml'")

if __name__ == "__main__":
    if not os.listdir('dataset'):
        print("[WARNING] Folder dataset kosong. Jalankan 'data_collector.py' dulu.")
    else:
        train_model()