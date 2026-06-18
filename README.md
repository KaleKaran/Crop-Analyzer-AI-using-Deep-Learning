# Crop Disease Analyzer Using Deep Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask"/>
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange.svg" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/CNN-Deep%20Learning-red.svg" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
</div>

<br/>

> A deep learning-based plant disease detection system that identifies crop leaf diseases from images and provides detailed disease documentation — built as a diploma final year capstone project.

**Institution:** Vivekanand Education Society's Polytechnic (VESP), Chembur  

**Guide:** Bindu Ramesh

---

## About

Crop Disease Analyzer is a CNN-based web application trained to detect plant leaf diseases from image input, with a Flask web interface for predictions and disease documentation.

The project was built to help farmers and agricultural workers identify crop diseases early, reducing crop loss through timely intervention.

---

## What It Does

- Upload a leaf image → model predicts the disease
- Displays disease name, confidence, and detailed documentation
- Supports multiple crops and disease classes
- Simple web interface built with Flask and HTML/CSS

---

## Diseases Detected

### Rhamnus Davurica

| Disease | Confidence Threshold | Documentation |
|---|---|---|
| Healthy | ≥ 75% | `Healthy Doc.docx` |
| Powdery Mildew | ≥ 75% | `Powdery Doc.docx` |
| Rust | ≥ 75% | `Rust Doc.docx` |

> If the model confidence is below 75%, the result is reported as **Uncertain**.

---

## Model Performance

### Confusion Matrix (Normalized)
![Confusion Matrix](screenshots/confusion_matrix.png)

| Class | Score |
|---|---|
| Powdery Mildew | 1.00 |
| Healthy | 0.65 |
| Rust | 0.51 |
| Background | 0.0 |

### F1-Confidence Curve
![F1 Curve](screenshots/f1_curve.png)

- Overall F1 score: **0.70 at confidence threshold 0.326**
- Best performing class: Powdery (~0.88 peak F1)

### Training & Validation Metrics
![Training Metrics](screenshots/training_metrics.png)

- Training loss (box, cls, dfl) consistently decreasing across 50 epochs
- Precision reaching ~0.80, Recall ~0.65
- mAP@50 steadily improving throughout training

---

## Screenshots

### Main Page
![Main Page](screenshots/main_page.png)

### After Predicting
![Prediction Result](screenshots/predict_result.png)

### Disease Documentation
![View Doc](screenshots/view_doc.png)

---

## Project Structure

```
Crop-Analyzer-AI-using-Deep-Learning-main/
│
├── Testing_DS/
│   └── link for testing dataset.txt      # Dataset download reference
│
├── model/
│   └── link for model.txt                # Trained model download reference
│
├── screenshots/                          # Model performance & UI screenshots
│
├── static/
│   ├── images/                           # UI assets
│   └── user uploaded/                    # User image uploads
│
├── templates/
│   ├── index.html                        # Landing page
│   ├── Analysis.html                     # Upload / analyze page
│   ├── predict.html                      # Prediction results page
│   ├── feedback.html                     # Feedback form
│   └── view_document.html               # Disease documentation viewer
│
├── Flask_Doc_DV.py                       # Main Flask application
├── Rhamnus_model2.ipynb                  # Model training notebook
├── Healthy Doc.docx                      # Healthy class documentation
├── Powdery Doc.docx                      # Powdery Mildew documentation
├── Rust Doc.docx                         # Rust disease documentation
└── README.md
```

---

## Tech Stack

| Layer | Stack |
|---|---|
| **ML Framework** | TensorFlow / Keras (CNN) |
| **Backend** | Python, Flask |
| **Frontend** | HTML, CSS (Jinja2 templates) |
| **Model Format** | Keras H5 (`.h5`) |
| **Dataset** | Plant Village / custom collected |

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/crop-disease-analyzer.git
cd crop-disease-analyzer
```

### 2. Download the model
Follow the link in `model/link for model.txt` and place `rahamnus.h5` inside the `model/` folder.

### 3. Install dependencies
```bash
pip install flask tensorflow keras numpy pillow python-docx matplotlib seaborn scikit-learn
```

### 4. Run the app
```bash
python Flask_Doc_DV.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## How It Works

1. User uploads a Rhamnus Davurica leaf image via the web interface
2. Flask receives the image and passes it to the trained CNN model (`rahamnus.h5`)
3. Model predicts the disease class with a confidence score
4. If confidence ≥ 75%, the result is displayed with disease documentation (symptoms, causes, treatment)
5. If confidence < 75%, the result is marked as **Uncertain**

---

## Model Training

The model was trained using the Jupyter notebook included in this repo:
- `Rhamnus_model2.ipynb` — data preprocessing, augmentation, CNN architecture (Conv2D, MaxPooling, BatchNormalization, Dropout), Adam optimizer, training and evaluation

---

## Origin

Built as the **final year capstone project** during Diploma in Automation & Robotics at VESP, Chembur. This project later served as the foundation for a published research paper presented at multiple Technical Paper Presentation (TPP) competitions.
