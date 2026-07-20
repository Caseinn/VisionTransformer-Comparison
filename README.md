# Tiny Vision Transformers pada CIFAR-10

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/timm-1.0.22-222222" alt="timm">
  <img src="https://img.shields.io/badge/dataset-CIFAR--10-4B8BBE" alt="CIFAR-10">
</p>

Eksperimen perbandingan DeiT-Tiny, EVA-02-Tiny, dan PiT-Tiny untuk klasifikasi gambar CIFAR-10. Model diambil dari `timm`, lalu dilatih pada pipeline yang sama agar hasil evaluasi lebih mudah dibandingkan.

Repository ini memakai notebook untuk menjalankan eksperimen dan modul Python di `src/` untuk logic yang dipakai berulang: konfigurasi model, data loader, training loop, evaluasi, inference timing, dan visualisasi.

## Latar Belakang

Vision Transformer sering dibahas dalam konteks model besar, tetapi varian kecilnya menarik untuk diuji pada dataset ringan seperti CIFAR-10. Tujuan repository ini bukan membuat benchmark final, melainkan menyediakan alur eksperimen yang rapi untuk melihat perilaku beberapa arsitektur transformer kecil pada task klasifikasi gambar.

Eksperimen ini menjawab beberapa pertanyaan praktis:

- seberapa baik model transformer kecil bekerja setelah training pada CIFAR-10;
- bagaimana perbandingan akurasi dan waktu inference antar model;
- seperti apa confusion matrix dan sample prediction yang dihasilkan;
- bagaimana patch importance dapat divisualisasikan untuk model yang mendukungnya.

> **Bukan benchmark resmi.** Hasil inference time bergantung pada hardware, versi PyTorch, driver, dan konfigurasi runtime. Angka di README sebaiknya dibaca sebagai hasil eksperimen lokal.

## Model yang Dibandingkan

| Model | Backbone `timm` | Pretrained Source | Checkpoint |
| --- | --- | --- | --- |
| DeiT-Tiny | `deit_tiny_distilled_patch16_224.fb_in1k` | ImageNet-1k | `deit_tiny_cifar10_best.pth` |
| EVA-02-Tiny | `eva02_tiny_patch14_224.mim_in22k` | ImageNet-22k | `eva02_tiny_cifar10_best.pth` |
| PiT-Tiny | `pit_ti_distilled_224.in1k` | ImageNet-1k | `pit_tiny_cifar10_best.pth` |

Semua model dikonfigurasi untuk 10 kelas CIFAR-10.

## Cara Pakai

### Prasyarat

- Python 3.10+
- Jupyter Notebook atau JupyterLab
- `pip`
- GPU CUDA opsional, tetapi disarankan untuk training

### Instalasi

Clone repository:

```bash
git clone https://github.com/Caseinn/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

`requirements.txt` memakai extra index PyTorch untuk wheel CUDA. Jika kamu menjalankan proyek ini di CPU-only environment atau butuh versi CUDA tertentu, sesuaikan instalasi PyTorch dengan environment lokal terlebih dahulu.

### Dataset

Dataset CIFAR-10 diambil dari `torchvision.datasets.CIFAR10` dan akan diunduh otomatis ke:

```text
./data
```

Transformasi gambar berada di `src/data.py`:

| Tahap | Penjelasan |
| --- | --- |
| Resize | Gambar CIFAR-10 diubah ke `224 x 224` agar cocok dengan input model pretrained. |
| Random horizontal flip | Dipakai saat training sebagai augmentasi ringan. |
| ToTensor | Mengubah gambar menjadi tensor PyTorch. |
| Normalize | Menggunakan mean dan standard deviation ImageNet. |

## Training

Jalankan notebook training sesuai model:

```text
train_deit_tiny.ipynb
train_eva02_tiny.ipynb
train_pit_tiny.ipynb
```

Setiap notebook training menjalankan alur berikut:

```text
CIFAR-10 -> Transform -> DataLoader -> timm Model -> Training -> Best Checkpoint
```

Checkpoint terbaik disimpan berdasarkan akurasi validasi:

```text
results/DeiT Tiny/deit_tiny_cifar10_best.pth
results/EVA-02 Tiny/eva02_tiny_cifar10_best.pth
results/PiT Tiny/pit_tiny_cifar10_best.pth
```

Secara default, training memakai:

| Komponen | Nilai |
| --- | --- |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` |
| Optimizer | `AdamW` |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Scheduler | `CosineAnnealingLR` |
| Epoch | `20` |

## Evaluasi

Jalankan notebook evaluasi sesuai model:

```text
test_deit_tiny.ipynb
test_eva02_tiny.ipynb
test_pit_tiny.ipynb
```

Notebook evaluasi akan:

- memuat checkpoint terbaik dari `results/`;
- melakukan prediksi pada test set CIFAR-10;
- menghitung accuracy, precision, recall, dan F1-score;
- menyimpan hasil prediksi ke CSV;
- membuat confusion matrix dan visualisasi sample prediction;
- membuat attention/patch importance map untuk model yang tersedia.

Format file prediksi:

```text
results/<nama-model>/predictions.csv
```

Kolom CSV:

| Kolom | Fungsi |
| --- | --- |
| `image_id` | Indeks gambar pada test set CIFAR-10. |
| `true_label` | Label sebenarnya. |
| `predicted_label` | Label hasil prediksi model. |

Contoh:

```csv
image_id,true_label,predicted_label
0,cat,cat
1,ship,ship
```

## Hasil Eksperimen

Ringkasan berikut berasal dari artifact yang tersedia di repository.

| Model | Params | Accuracy | Inference Time | Best Checkpoint |
| --- | ---: | ---: | ---: | --- |
| DeiT-Tiny | ~5.5M | 91.89% | 0.915 ms/img | `results/DeiT Tiny/deit_tiny_cifar10_best.pth` |
| EVA-02-Tiny | ~5.5M | 92.81% | 1.254 ms/img | `results/EVA-02 Tiny/eva02_tiny_cifar10_best.pth` |
| PiT-Tiny | ~4.5M | 92.01% | 0.887 ms/img | `results/PiT Tiny/pit_tiny_cifar10_best.pth` |

## Visualisasi

### DeiT-Tiny

Kurva training dan validasi:

![Kurva DeiT-Tiny](results/DeiT%20Tiny/curves.png)

Confusion matrix:

![Confusion matrix DeiT-Tiny](results/DeiT%20Tiny/confusion.png)

Attention map:

![Attention map DeiT-Tiny](results/DeiT%20Tiny/attention.png)

Sample prediction:

![Sample prediction DeiT-Tiny](results/DeiT%20Tiny/samples.png)

### EVA-02-Tiny

Kurva training dan validasi:

![Kurva EVA-02-Tiny](results/EVA-02%20Tiny/curves.png)

Confusion matrix:

![Confusion matrix EVA-02-Tiny](results/EVA-02%20Tiny/confusion.png)

Attention map:

![Attention map EVA-02-Tiny](results/EVA-02%20Tiny/attention.png)

Sample prediction:

![Sample prediction EVA-02-Tiny](results/EVA-02%20Tiny/samples.png)

### PiT-Tiny

Kurva training dan validasi:

![Kurva PiT-Tiny](results/PiT%20Tiny/curves.png)

Confusion matrix:

![Confusion matrix PiT-Tiny](results/PiT%20Tiny/confusion.png)

Sample prediction:

![Sample prediction PiT-Tiny](results/PiT%20Tiny/samples.png)

## Struktur Proyek

```text
.
|-- data/
|   `-- cifar-10-batches-py/        # Dataset CIFAR-10 lokal
|-- results/
|   |-- DeiT Tiny/                  # Checkpoint dan hasil evaluasi DeiT-Tiny
|   |-- EVA-02 Tiny/                # Checkpoint dan hasil evaluasi EVA-02-Tiny
|   `-- PiT Tiny/                   # Checkpoint dan hasil evaluasi PiT-Tiny
|-- src/
|   |-- __init__.py
|   |-- config.py                   # Registry konfigurasi model
|   |-- data.py                     # Transform dan DataLoader CIFAR-10
|   |-- evaluate.py                 # Prediksi, metrik, dan ekspor CSV
|   |-- inference.py                # Pengukuran inference time
|   |-- model.py                    # Pembuatan model dan ringkasan parameter
|   |-- train.py                    # Training loop dan validasi
|   `-- visualize.py                # Plot hasil eksperimen
|-- test_deit_tiny.ipynb
|-- test_eva02_tiny.ipynb
|-- test_pit_tiny.ipynb
|-- train_deit_tiny.ipynb
|-- train_eva02_tiny.ipynb
|-- train_pit_tiny.ipynb
|-- requirements.txt
`-- README.md
```

## Batasan

Beberapa batasan eksperimen ini:

| Batasan | Dampak |
| --- | --- |
| Dataset hanya CIFAR-10 | Hasil tidak langsung mewakili performa pada dataset gambar yang lebih kompleks. |
| Input di-resize ke `224 x 224` | Ada perubahan skala besar dari resolusi asli CIFAR-10. |
| Training berfokus pada klasifikasi 10 kelas | Repository ini tidak mencakup detection, segmentation, atau task vision lain. |
| Inference time bersifat lokal | Angka dapat berubah pada hardware dan environment berbeda. |
| Artifact hasil training ikut disimpan | Ukuran repository dapat bertambah karena checkpoint dan gambar hasil evaluasi. |
