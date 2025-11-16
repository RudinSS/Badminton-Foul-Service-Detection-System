# 🏸 Sistem Deteksi Pelanggaran Servis Bulu Tangkis

Sistem deteksi objek secara real-time yang menggunakan model YOLOv5 untuk mendeteksi pelanggaran servis dalam permainan bulu tangkis. Aplikasi ini memonitor servis pemain dan memberikan peringatan suara ("FOUL") jika servis terdeteksi dilakukan di atas ambang batas ketinggian yang telah ditentukan.

## 📋 Deskripsi

Proyek ini menggunakan YOLOv5 yang telah dilatih secara khusus untuk mendeteksi dua objek utama:
- **Racket** (raket)
- **Shuttle** (kok)

Logika inti, yang terdapat dalam `detect.py`, memeriksa interaksi antara kedua objek ini. Jika bounding box raket dan kok terdeteksi tumpang tindih (menandakan pukulan), skrip akan memeriksa posisi vertikal (koordinat y) dari kok. Jika pukulan terjadi di atas garis batas tinggi yang ditentukan pengguna, sistem akan menandainya sebagai **"FOUL"** dan memutar suara peringatan.

Proyek ini dilengkapi dengan:
- **GUI** sederhana menggunakan Tkinter (`gui.py`) untuk kemudahan penggunaan
- **CLI** standar (`detect.py`) untuk deteksi via command-line

## 🎯 Fitur

- ✅ **Deteksi Real-time**: Menganalisis umpan video langsung dari webcam atau file video
- 🚨 **Deteksi Pelanggaran Servis**: Secara otomatis mendeteksi jika kok dipukul di atas ketinggian servis yang sah
- 🔊 **Peringatan Suara**: Memutar suara peringatan (`winsound.Beep`) ketika pelanggaran terdeteksi
- ⚙️ **Ambang Batas yang Dapat Disesuaikan**:
  - **Ambang Batas Ketinggian**: Garis pelanggaran (dalam piksel) dapat diatur melalui GUI
  - **Ambang Batas Keyakinan**: Menyesuaikan sensitivitas deteksi model (dalam %)
- 🖥️ **Antarmuka Ganda**: Dapat dijalankan melalui GUI yang ramah pengguna atau CLI

## 🛠️ Instalasi

### 1. Clone Repositori

```bash
git clone https://github.com/username/Badminton-Foul-Service-Detection-System.git
cd Badminton-Foul-Service-Detection-System
```

### 2. Instal Dependensi

Proyek ini memiliki beberapa dependensi Python yang tercantum dalam `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Unduh Bobot Model (Weights)

Skrip deteksi mengharapkan file bobot model yang telah dilatih bernama `best.pt`. Pastikan Anda menempatkan file `best.pt` di direktori utama proyek.

## 🚀 Penggunaan

Ada dua cara untuk menjalankan deteksi:

### 1. Menggunakan GUI (Disarankan)

Jalankan skrip `gui.py` untuk membuka antarmuka grafis:

```bash
python gui.py
```

Antarmuka ini memungkinkan Anda untuk mengonfigurasi parameter deteksi dengan mudah:

| Parameter | Deskripsi |
|-----------|-----------|
| **Height Threshold (px)** | Atur ketinggian garis pelanggaran dalam piksel |
| **Select Camera** | Pilih indeks kamera/webcam yang akan digunakan (misalnya, 0 untuk webcam internal) |
| **Confidence Threshold (%)** | Atur tingkat keyakinan minimum (0-100) untuk deteksi objek |
| **Run Detection** | Memulai jendela deteksi real-time |

### 2. Menggunakan Command-Line (CLI)

Anda dapat menjalankan deteksi langsung dari terminal menggunakan `detect.py`.

**Contoh Perintah** (menggunakan webcam):

```bash
python detect.py --weights best.pt --source 0 --conf-thres 0.3
```

#### Argumen Penting:

- `--weights`: (Wajib) Path ke file bobot model Anda (misalnya, `best.pt`)
- `--source`: Sumber video. Bisa berupa:
  - Indeks webcam (`0`, `1`, ...)
  - File video (`video.mp4`)
  - URL stream
- `--conf-thres`: Ambang batas keyakinan (misalnya, `0.3` untuk 30%)

> **⚠️ Catatan**: Saat menggunakan CLI, ambang batas ketinggian diatur secara default di dalam skrip `detect.py`. Untuk mengubahnya, Anda harus mengedit nilai `height_threshold` langsung di dalam fungsi `run` pada file `detect.py`. GUI adalah cara yang disarankan jika Anda perlu sering mengubah nilai ini.

## 📁 Struktur Proyek

```
.
├── detect.py         # Skrip deteksi utama (CLI) dengan logika deteksi pelanggaran
├── gui.py            # Aplikasi GUI (Tkinter) untuk menjalankan deteksi
├── requirements.txt  # Daftar dependensi Python
├── best.pt           # (WAJIB ADA) File bobot model YOLOv5 yang telah dilatih
├── models/           # Definisi model YOLOv5
├── utils/            # Skrip utilitas YOLOv5
└── *.yaml            # File konfigurasi model YOLOv5
```


