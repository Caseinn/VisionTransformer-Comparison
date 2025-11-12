---

# Tiny Vision Transformers on CIFAR-10

*DeiT-Tiny • EVA-02-Tiny • PiT-Tiny*

This repo contains training & evaluation notebooks for several “tiny” transformer variants on CIFAR. Each model folder stores artifacts (best checkpoint, predictions, curves, confusion matrix, attention maps, and sample grids).

```
|
├─ DeiT Tiny/
├─ EVA-02 Tiny/
├─ PiT Tiny/
├─ data/
├─ requirements.txt
├─ train_deit_tiny.ipynb   ├─ test_deit_tiny.ipynb
├─ train_eva02_tiny.ipynb  ├─ test_eva02_tiny.ipynb
├─ train_pit_tiny.ipynb    ├─ test_pit_tiny.ipynb
```

## 0) Get the code

```bash
git clone https://github.com/Caseinn/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison
```

## 1) Environment

```bash
# Python 3.10+ recommended
python -m venv .venv
# or: conda create -n tiny-vit python=3.10

# activate
source .venv/bin/activate         # macOS/Linux
# .venv\Scripts\activate          # Windows

# install deps
pip install -r requirements.txt

# optional: speedups
pip install --upgrade torch torchvision timm
```

## 2) Data

By default the notebooks will download CIFAR-10 automatically into `./data`.
You can pre-download if you prefer and keep the same path.

## 3) Train

Open the notebook for the model you want and run all cells:

* `train_deit_tiny.ipynb`
* `train_eva02_tiny.ipynb`
* `train_pit_tiny.ipynb`

Each training notebook will:

* build the model from `timm`
* fine-tune on CIFAR
* save the best checkpoint into the corresponding model folder (e.g., `DeiT Tiny/deit_tiny_cifar10_best.pth`)

## 4) Test / Inference

Use the paired **test** notebooks:

* `test_deit_tiny.ipynb`
* `test_eva02_tiny.ipynb`
* `test_pit_tiny.ipynb`

They will:

* load the best checkpoint from the model folder
* evaluate on the test set
* export:

  * `predictions.csv` (per-image predicted vs true label + score)
  * `confusion.png` (confusion matrix)
  * `curves.png` (loss/accuracy curves if provided)
  * `attention.png` (representative attention visualization if available)
  * `samples.png` (correct/incorrect sample grid)

## 5) Results & Figures

After running tests, you’ll have images inside each model’s folder.

### CIFAR-10 — Qualitative & Diagnostics

**DeiT-Tiny**

* Training/validation curves
  ![DeiT curves](DeiT%20Tiny/curves.png)
* Confusion matrix
  ![DeiT confusion](DeiT%20Tiny/confusion.png)
* Attention visualization
  ![DeiT attention](DeiT%20Tiny/attention.png)
* Sample predictions
  ![DeiT samples](DeiT%20Tiny/samples.png)

**EVA-02-Tiny**

* Training/validation curves
  ![EVA02 curves](EVA-02%20Tiny/curves.png)
* Confusion matrix
  ![EVA02 confusion](EVA-02%20Tiny/confusion.png)
* Attention visualization
  ![EVA02 attention](EVA-02%20Tiny/attention.png)
* Sample predictions
  ![EVA02 samples](EVA-02%20Tiny/samples.png)

**PiT-Tiny**

* Training/validation curves
  ![PiT curves](PiT%20Tiny/curves.png)
* Confusion matrix
  ![PiT confusion](PiT%20Tiny/confusion.png)
* Sample predictions
  ![PiT samples](PiT%20Tiny/samples.png)

### CIFAR-10 — Summary Table

| Model       | Params (M) | Acc (%) | Inference Time (ms/img) | Best CKPT                                 |
| ----------- | ---------- | ------- | ----------------------- | ----------------------------------------- |
| DeiT-Tiny   | ~5.5       | 91.89   | 0.915                   | `DeiT Tiny/deit_tiny_cifar10_best.pth`    |
| EVA-02-Tiny | ~5.5       | 92.81   | 1.254                   | `EVA-02 Tiny/eva02_tiny_cifar10_best.pth` |
| PiT-Tiny    | ~4.5       | 92.01   | 0.887                   | `PiT Tiny/pit_tiny_cifar10_best.pth`      |
