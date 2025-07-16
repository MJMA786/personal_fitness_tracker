# 🔥 Personal Fitness Tracker

A machine learning-powered Streamlit web app that predicts the number of calories burned during exercise based on personal and activity-related parameters like age, gender, BMI, heart rate, body temperature, and exercise duration.

---

## 📌 Project Overview

This app allows users to input their fitness data and instantly receive an estimate of calories burned. It uses a trained Random Forest Regressor model and provides insightful feedback comparing the user's data with others from a real-world dataset.

---

## ✨ Features

- 🔢 **BMI Calculation** from height and weight
- ⚙️ **Random Forest** model for calorie prediction
- 🧠 **Real-time feedback** on user performance
- 📊 **Comparison with dataset** averages (e.g., heart rate, age)
- 🔍 Shows **similar user results** based on predicted calories
- 💾 Model is **cached** and reused for fast predictions

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (RandomForestRegressor)
- Streamlit (web interface)
- Joblib (model saving/loading)

---

## 📁 File Structure

```bash
.
├── app.py                      # Main Streamlit application
├── rf_model_train.ipynb        # Notebook to train and save model
├── fitness_tracker.ipynb       # Analysis and testing notebook
├── random_forest_calories.pkl  # Trained Random Forest model
├── calories.csv                # Calories data
├── exercise.csv                # Exercise session data
└── README.md                   # This file
└── License                     # license
```

---

## 🚀 How to Run the App
## Clone the repository:
- git clone https://github.com/MJMA786/personal_fitness_tracker.git
- cd personal-fitness-tracker

---

## Install dependencies:

- pip install -r requirements.txt

---

## Run the Streamlit app:

- streamlit run app.py

---
## 🧠 Dataset Info

- exercise.csv: Contains physical metrics like duration, heart rate, body temp, height, and weight.
- calories.csv: Calories burned per session.
Merged and processed for training the model.

---

## 📊 Example Prediction Insight
- 🔥 132.54 kilocalories burned
- ✅ You are in the healthy activity range, similar to about 70% of users.

---

## 📄 License
- This project is licensed under the MIT License.
