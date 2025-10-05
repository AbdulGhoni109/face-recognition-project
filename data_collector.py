# data_collector.py
import cv2
import os

def collect_dataset():
    dataset_path = "dataset/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    person_id = input("Masukkan ID untuk orang ini (angka, contoh: 1): ")
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

    print("\n[INFO] Pengambilan dataset selesai.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_dataset()