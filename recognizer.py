# recognizer.py
import cv2
import json
import os

def start_recognition():
    # --- Memuat file data nama ---
    names_file = 'names.json'
    if not os.path.exists(names_file):
        print(f"[ERROR] File '{names_file}' tidak ditemukan. Jalankan pengambilan dataset dulu.")
        return
        
    with open(names_file, 'r') as f:
        names_data = json.load(f)

    # --- Inisialisasi Recognizer dan Kamera ---
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('models/face-model.yml') 
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cap = cv2.VideoCapture(0)
    print("\n[INFO] Memulai kamera. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            person_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:
                # Ambil nama dari data JSON menggunakan ID yang terdeteksi
                # Kita ubah ID (int) ke string karena key di JSON adalah string
                display_name = names_data.get(str(person_id), "ID Tidak Dikenal")
                confidence_text = f" {round(100 - confidence)}%"
            else:
                display_name = "Unknown"
                confidence_text = f" {round(100 - confidence)}%"
            
            cv2.putText(frame, str(display_name), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence_text), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        cv2.imshow('Pengenalan Wajah', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    print("\n[INFO] Keluar dari program.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists('models/face-model.yml'):
        print("[WARNING] File model tidak ditemukan. Jalankan 'trainer.py' dulu.")
    else:
        start_recognition()