# data_collector.py
import cv2
import os
import json

def collect_dataset():
    # --- Inisialisasi ---
    dataset_path = "dataset/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    
    # --- Memuat atau membuat file data nama ---
    names_file = 'names.json'
    
    # KODE YANG DIPERBAIKI: Cek apakah file ada DAN tidak kosong
    if os.path.exists(names_file) and os.path.getsize(names_file) > 0:
        with open(names_file, 'r') as f:
            names_data = json.load(f)
    else:
        # Jika file tidak ada atau kosong, mulai dengan dictionary kosong
        names_data = {}

    # --- Meminta Input ID dan Nama dari Pengguna ---
    person_id = input("Masukkan ID (angka, contoh: 1): ")
    if person_id in names_data:
        print(f"[ERROR] ID {person_id} sudah digunakan untuk nama '{names_data[person_id]}'. Gunakan ID lain.")
        return
        
    person_name = input(f"Masukkan Nama untuk ID {person_id}: ")

    print("\n[INFO] Memulai pengambilan gambar. Lihat ke kamera dan tunggu...")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengakses kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"{dataset_path}Person.{person_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.imshow("Pengambilan Dataset", frame)

        if count >= 30:
            break
        elif cv2.waitKey(1) == ord('q'):
            break

    # --- Simpan nama baru ke file JSON ---
    names_data[person_id] = person_name
    with open(names_file, 'w') as f:
        json.dump(names_data, f, indent=4)
    print(f"\n[INFO] Data untuk ID {person_id} ({person_name}) berhasil disimpan.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_dataset()