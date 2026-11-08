# AI or Real Photo Classifier

Proyek klasifikasi gambar untuk membedakan foto nyata dan gambar yang dibuat oleh AI. Model dapat mengenali dua jenis objek: **manusia** dan **pemandangan**.

## Demo Streamlit

Repository ini menyediakan aplikasi demo melalui [`app.py`](app.py). Aplikasi memuat model `image_classification_cnn.h5`, menerima unggahan gambar, lalu menjalankan langkah yang sama seperti notebook: mengonversi gambar ke RGB, mengubah ukuran menjadi 150 × 150 piksel, dan menormalisasi nilai piksel ke rentang 0–1 sebelum prediksi.

Jalankan secara lokal:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Untuk deployment, pilih `app.py` sebagai entrypoint di Streamlit Community Cloud. Model `.h5` harus tetap berada di root repository bersama file aplikasi.

## Yang dapat diklasifikasikan

Model menghasilkan satu dari empat label berikut:

| Label | Arti |
| --- | --- |
| `ManusiaAI` | Gambar manusia buatan AI |
| `ManusiaReal` | Foto manusia nyata |
| `PemandanganAI` | Gambar pemandangan buatan AI |
| `PemandanganReal` | Foto pemandangan nyata |

## Gambaran proses

```text
Gambar asli
    -> augmentasi gambar
    -> pembagian data latih dan pengujian
    -> pelatihan Convolutional Neural Network (CNN)
    -> model image_classification_cnn.h5
```

Dataset awal terdiri dari 1.000 gambar: 250 gambar untuk masing-masing kelas. Setiap gambar dibuat menjadi lima variasi melalui augmentasi, sehingga menghasilkan 5.000 gambar tambahan. Data yang digunakan untuk pelatihan dan pengujian berjumlah 6.000 gambar karena menggabungkan 1.000 gambar asli dan 5.000 gambar hasil augmentasi.

| Lokasi | Isi | Jumlah |
| --- | --- | ---: |
| `dataset_image/` | Dataset asli | 1.000 gambar |
| `augmented_image/` | Hasil augmentasi | 5.000 gambar |
| `image_machine_learning/train/` | Data pelatihan | 4.800 gambar |
| `image_machine_learning/testing/` | Data pengujian | 1.200 gambar |

## Model

Model utama memakai CNN sederhana dengan tiga tahap ekstraksi fitur gambar, kemudian lapisan klasifikasi untuk memilih satu dari empat kategori. Gambar diproses pada ukuran **150 × 150 piksel**.

Konfigurasi utama:

- 20 epoch pelatihan
- Batch size 32
- Optimizer Adam
- Klasifikasi multikelas dengan softmax
- Dropout untuk membantu mengurangi overfitting

Hasil eksperimen terakhir yang tersimpan pada notebook menunjukkan akurasi pengujian sekitar **85,08%**.

> Catatan evaluasi: pembagian data tampak dilakukan setelah proses augmentasi. Artinya, variasi dari gambar sumber yang sama mungkin muncul pada data latih dan data uji. Untuk pengukuran yang lebih ketat pada pengembangan berikutnya, pisahkan gambar asli menjadi train/test terlebih dahulu, lalu lakukan augmentasi hanya pada data train.

## Struktur repository

```text
.
├── CNN.ipynb                         # Notebook utama untuk melatih dan mengevaluasi CNN
├── augmented_image.ipynb             # Notebook utama untuk membuat data augmentasi
├── image_classification_cnn.h5       # Model hasil pelatihan
├── dataset_image/                    # Gambar asli per kelas
├── augmented_image/                  # Gambar hasil augmentasi per kelas
├── image_machine_learning/
│   ├── train/                        # Dataset untuk pelatihan
│   └── testing/                      # Dataset untuk pengujian
├── NyobaAugmentasi/                  # Eksperimen augmentasi awal
└── web/                              # Prototipe antarmuka web PHP
```

## Notebook utama

### `augmented_image.ipynb`

Membuat lima variasi dari setiap gambar asli menggunakan Albumentations dan OpenCV:

1. Horizontal flip
2. Vertical flip
3. Perubahan brightness dan contrast
4. Gaussian blur
5. Rotasi

### `CNN.ipynb`

Memuat data train dan test, melatih model CNN, mengevaluasi hasilnya, menampilkan grafik akurasi/loss, lalu menyimpan model ke `image_classification_cnn.h5`.

## Catatan penggunaan

Path dataset di beberapa notebook masih mengarah ke path lokal Windows milik pembuat proyek. Sesuaikan variabel path tersebut dengan lokasi repository di komputer Anda sebelum menjalankan notebook.

Folder `web/` adalah prototipe awal berbasis PHP. Model dan notebook merupakan bagian yang paling siap digunakan dari repository ini.

## Teknologi

- Python
- TensorFlow / Keras
- OpenCV
- Albumentations
- NumPy dan Matplotlib
- Jupyter Notebook

## Rencana pengembangan

- Menyediakan aplikasi demo publik dengan Streamlit
- Menambah skrip inferensi yang dapat digunakan ulang
- Membuat pembagian data yang mencegah kebocoran antarvariasi gambar
- Menambahkan laporan metrik per kelas dan confusion matrix
