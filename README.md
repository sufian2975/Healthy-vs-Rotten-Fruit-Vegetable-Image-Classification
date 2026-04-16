# 🍎 Healthy vs Rotten Fruit & Vegetable Classification

## 📌 Overview

**Healthy-vs-Rotten-Fruit-Vegetable-Image-Classification** is a computer vision project that classifies fruits and vegetables as **fresh (healthy)** or **spoiled (rotten)** using deep learning techniques.

---

## 🎯 Key Features

* Machine Learning / Computer Vision project
* Classifies fruits and vegetables as **Healthy** or **Rotten**
* Built using **TensorFlow / Keras**
* Uses labeled image dataset
* Outputs prediction with confidence score

---

## 🧠 Model & Workflow

The project follows a standard deep learning pipeline:

1. Data preprocessing
2. Image augmentation
3. Model training (CNN)
4. Model evaluation
5. Prediction

### 🔍 Architectures Used

* **Custom Convolutional Neural Network (CNN)**
* **EfficientNetB0 (Transfer Learning)**

The model analyzes features like **color, texture, and shape** to determine freshness.

---

## 📂 Project Structure

```
.
├── app.py
├── fruit-disease.ipynb
├── README.md
```

---

## 📊 Applications

* Supermarkets (automated quality control)
* Warehouses (sorting systems)
* Smart agriculture
* Food waste reduction

---

## ⚙️ Installation

```
git clone https://github.com/sufian2975/Healthy-vs-Rotten-Fruit-Vegetable-Image-Classification.git
cd Healthy-vs-Rotten-Fruit-Vegetable-Image-Classification
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the script:

```
python app.py
```

Or open the notebook:

```
fruit-disease.ipynb
```

---

## 📈 Results

The model classifies images into:

* Healthy
* Rotten

Performance depends on dataset quality and training.

---

## 🔮 Future Improvements

* Deploy as a web app (Streamlit / Flask)
* Improve accuracy with more data
* Add real-time detection


