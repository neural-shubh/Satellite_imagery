# Satellite Image Classification

A deep learning project that classifies satellite images into 4 land cover categories using **MobileNetV2 Transfer Learning**, achieving **~99.3% validation accuracy**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.3%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Classes

| Class | Description |
|---|---|
| ☁️ Cloudy | Cloud-covered regions |
| 🏜️ Desert | Arid and sandy terrain |
| 🌿 Green Area | Forests and vegetation |
| 💧 Water | Oceans, lakes, rivers |

---

## 📁 Project Structure

```
satellite-classification/
│
├── satellite_classification.ipynb   # Main training notebook
├── satellite_classifier_final.keras # Saved model
├── class_indices.json               # Class label mappings
├── model_info.json                  # Model metadata
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
│
└── extracted_data/
    ├── train/
    │   ├── cloudy/
    │   ├── desert/
    │   ├── green_area/
    │   └── water/
    └── val/
        ├── cloudy/
        ├── desert/
        ├── green_area/
        └── water/
```

---

## 📊 Dataset

| Split | Cloudy | Desert | Green Area | Water | Total |
|---|---|---|---|---|---|
| Train | 1,447 | 1,085 | 1,449 | 1,439 | 5,420 |
| Val | 547 | 406 | 549 | 539 | 2,041 |
| **Total** | **1,994** | **1,491** | **1,998** | **1,978** | **7,461** |

---

## 🧠 Model Architecture

```
MobileNetV2 (pretrained on ImageNet, frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(128, relu)
    ↓
Dropout(0.5)
    ↓
Dense(4, softmax)
```

### Training Strategy
- **Phase 1** — Base frozen, train top layers for 15 epochs (lr=1e-3)
- **Phase 2** — Unfreeze last 20 layers, fine-tune for 10 epochs (lr=1e-5)

---

## 📈 Results

| Metric | Value |
|---|---|
| Final Val Accuracy | **99.3%** |
| Final Val Loss | **0.0189** |
| Final Train Accuracy | 96.7% |

### Training Curve
> Train both phases and plot with `plot_history(history1, history2)`

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/satellite-classification.git
cd satellite-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset
Place your `satellite_dataset.zip` in the project root, then run:
```python
import zipfile
with zipfile.ZipFile('satellite_dataset.zip', 'r') as z:
    z.extractall('extracted_data/data')
```

Then run the train/val split cell in the notebook.

### 4. Run the notebook
Open `satellite_classification.ipynb` in Google Colab or Jupyter and run all cells.

---

## 🔍 Inference on New Images

```python
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model and class names
model = load_model('satellite_classifier_final.keras')
with open('class_indices.json') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    print(f"Prediction : {class_names[idx]}")
    print(f"Confidence : {preds[0][idx]*100:.2f}%")

predict('your_image.jpg')
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras 2.x**
- **MobileNetV2** (ImageNet pretrained)
- **NumPy, Matplotlib, Seaborn**
- **Scikit-learn**
- **Google Colab** (training environment)

---

## 📋 Requirements

See `requirements.txt` for full list.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Shubh Sharma**  
