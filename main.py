# main.py
import os
from data_collector import collect_dataset
from trainer import train_model
from recognizer import start_recognition

def main_menu():
    while True:
        print("\n" + "="*30)
        print("   MENU UTAMA FACE RECOGNITION")
        print("="*30)
        print("1. Ambil Dataset Wajah Baru")
        print("2. Latih Model")
        print("3. Jalankan Pengenalan Wajah")
        print("4. Keluar")
        print("="*30)
        
        choice = input("Pilih opsi [1/2/3/4]: ")

        if choice == '1':
            collect_dataset()
        elif choice == '2':
            if not os.listdir('dataset'):
                print("\n[PERINGATAN] Folder 'dataset' kosong. Jalankan Opsi 1 dulu.")
            else:
                train_model()
        elif choice == '3':
            if not os.path.exists('models/face-model.yml'):
                print("\n[PERINGATAN] Model belum dilatih. Jalankan Opsi 2 dulu.")
            else:
                start_recognition()
        elif choice == '4':
            print("\nTerima kasih! Sampai jumpa. 👋")
            break
        else:
            print("\nPilihan tidak valid. Silakan masukkan angka 1, 2, 3, atau 4.")

if __name__ == "__main__":
    main_menu()