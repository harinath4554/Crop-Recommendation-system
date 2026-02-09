# Crop Recommendation System Using Machine Learning

## 📘 Overview
The Crop Recommendation System is a machine learning–based application designed to recommend the most suitable crop for cultivation based on soil nutrients and environmental conditions. This project supports data-driven decision-making in agriculture.

---

## 🎯 Objectives
- Recommend suitable crops based on soil and climate parameters  
- Apply machine learning techniques to agricultural data  
- Provide a simple and accessible prediction system  

---

## 🧠 Methodology
1. Load and preprocess agricultural dataset  
2. Train a supervised machine learning classification model  
3. Save the trained model using Pickle  
4. Use a Flask web application for prediction  

---

## 📂 Repository Structure
'''Crop-Recommendation-system/
│
├── Crop_recommendation.csv
├── Crop_recommendation.ipynb
├── app.py
├── model.pkl
└── requirements.txt
'''



---

## 📊 Dataset Description

| Feature | Description |
|-------|-------------|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| Temperature | Temperature (°C) |
| Humidity | Relative humidity (%) |
| pH | Soil pH value |
| Rainfall | Rainfall (mm) |
| Label | Recommended crop |

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Flask  
- Jupyter Notebook  

---
🚀 Future Enhancements

Integration with real-time weather APIs

Fertilizer recommendation module

Support for regional crop variations

Cloud deployment for wider accessibility

Model comparison and performance optimization
