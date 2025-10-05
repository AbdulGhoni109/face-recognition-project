# recognizer.py
import cv2

def start_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('models/face-model.yml') 
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # TODO: SESUAIKAN DAFTAR NAMA INI DENGAN ID ANDA
    names = ['Unknown', 'Nama Anda']  # names[1] untuk ID 1, names[2] untuk ID 2, dst.

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
                display_name = names[person_id] if person_id < len(names) else "ID Tidak Dikenal"
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