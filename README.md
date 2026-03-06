# 🛡️ SENTINEL — Intelligent Media Forgery Detection

> **6th Semester CSEDS Project** — Multimodal AI system for detecting image forgeries using Computer Vision + NLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-ff4b4b)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 What This Project Does

SENTINEL detects whether an image has been **forged, AI-generated, or manipulated** using a three-layer detection pipeline:

| Layer | Technology | Detects |
|---|---|---|
| L1 — SBI Splice Detector | EfficientNet-B4 + DistilBERT | Copy-paste & region splicing |
| L2 — CLIP Semantic Engine | CLIP ViT-B/32 (HuggingFace) | AI-generated images (GAN, Diffusion, MidJourney, DALL-E) |
| L3 — DCT Frequency Analyzer | Discrete Cosine Transform | Unnatural frequency artifacts |
| L4 — Caption Consistency | CLIP multi-probe scoring | Image-text semantic mismatch |

---

## 🏆 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **93.1%** |
| F1 Score | **0.9285** |
| AUC-ROC | **0.9798** |
| Best Val F1 | **0.9425** |
| Best Val Accuracy | **94.44%** |
| Optimal Epoch | **14 / 15** |

---

## 🧠 Architecture

```
Image ──→ EfficientNet-B4 ──→ Spatial Features (7×7) ──→ U-Net Decoder ──→ Heatmap (224×224)
                         └──→ Pooled Features (1792) ──→
                                                         Cross-Attention Fusion ──→ Fake/Real
Caption ──→ DistilBERT ──→ CLS Embedding (768) ──→
```

- **Vision Branch**: EfficientNet-B4 pretrained on ImageNet, last 2 blocks fine-tuned
- **Text Branch**: DistilBERT-base-uncased, last transformer block fine-tuned
- **Fusion**: 8-head Cross-Attention (512-dim) — vision queries text for semantic contradictions
- **Localization**: U-Net decoder with 5 upsampling blocks (1792→512→256→128→64→32→1)
- **Total Parameters**: 99.8M | **Trainable**: 36.9M

---

## 📦 Dataset

- **Flickr8k** — 8,091 real photographs with human-written captions
- **SBI Augmentation** — Self-Blended Image forgery generation (on-the-fly)
- **Caption Perturbation** — Random caption swap/shuffle for fake samples
- **Split**: 70% Train / 15% Val / 15% Test

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or 3.11 recommended (3.13 has known CLIP compatibility issues)
- 4GB RAM minimum
- GPU optional (CPU works fine for inference)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOURUSERNAME/sentinel-forgery-detection.git
cd sentinel-forgery-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### If on Python 3.13
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 🖥️ Usage

1. Open the app at `http://localhost:8501`
2. Upload any image (JPG, PNG, WEBP)
3. Optionally enter a caption describing the image
4. Click **▶ INITIATE FORENSIC ANALYSIS**
5. View:
   - **Verdict**: Authentic / AI Generated / Spliced / Manipulated
   - **Layer-by-layer scores** from all 3 detection layers
   - **Localization heatmap** showing suspected forged regions
   - **Caption consistency analysis** (if caption provided)

---

## 📁 Project Structure

```
sentinel-forgery-detection/
├── app.py                  # Streamlit web application (UI + inference)
├── requirements.txt        # All dependencies
├── best_model.pth          # Trained model weights (download separately)
├── README.md               # This file
└── model/
    └── architecture.py     # Full model definition (EfficientNet + DistilBERT + U-Net)
```

---

## 🔧 Training (Optional — Reproduce Results)

Training was done on **Google Colab (T4 GPU)**. To retrain:

1. Open `ForgeryDetection_MasterNotebook.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Run all cells top to bottom
4. Training takes ~2-3 hours for 15 epochs
5. Best weights auto-saved to Google Drive

**Training Configuration:**
```
Optimizer  : AdamW (lr=2e-4, weight_decay=1e-4)
Scheduler  : CosineAnnealingLR (T_max=15, eta_min=1e-6)
Loss       : Focal Loss (γ=2) + Dice Loss × 0.5
Batch size : 8 (gradient accumulation × 4 = effective batch 32)
Image size : 224×224
Epochs     : 15 (early stopping patience=5)
```

---

## 📊 Results Visualization

| Real Image | Forged Image | Ground Truth Mask | Model Heatmap |
|---|---|---|---|
| Original photo | SBI splice applied | White = forged region | Red = model prediction |

*The model correctly localizes forged regions with IoU ~0.70*

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Deep Learning | PyTorch 2.1 |
| Vision Model | EfficientNet-B4 (timm) |
| Language Model | DistilBERT (HuggingFace Transformers) |
| AI Detection | CLIP ViT-B/32 (HuggingFace) |
| Segmentation | U-Net Decoder |
| Image Processing | OpenCV, PIL, Albumentations |
| Frequency Analysis | SciPy DCT |
| Web Interface | Streamlit |
| Training Platform | Google Colab (T4 GPU) |

---

## 👨‍💻 Author

**Your Name**
6th Semester — Computer Science & Engineering (Data Science)
*[Your College Name]*

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) — Adityajn105 on Kaggle
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [DistilBERT](https://arxiv.org/abs/1910.01108) — Sanh et al., 2019
- [CLIP](https://arxiv.org/abs/2103.00020) — Radford et al., OpenAI 2021
- [SBI Method](https://arxiv.org/abs/2202.10870) — Shiohara & Yamasaki, 2022
