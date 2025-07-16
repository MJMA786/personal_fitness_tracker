import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# Title
st.write("## Personal Fitness Tracker")
st.write("In this WebApp, you can predict your burned calories based on parameters such as Age, Gender, BMI, etc.")

st.sidebar.header("User Input Parameters: ")

# Function to get user input
def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    # Match feature names to the training dataset
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Encoded as 1 for male, 0 for female
    }

    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
st.write(df)

# Load dataset
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    
    # Calculate BMI
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = round(df["BMI"], 2)
    
    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    return df

exercise_df = load_data()

# Train or load the model
@st.cache_resource
def train_or_load_model():
    try:
        return joblib.load("random_forest_calories.pkl")
    except:
        # Train the model if it doesn't exist
        X = exercise_df.drop("Calories", axis=1)
        y = exercise_df["Calories"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=1)
        model.fit(X_train, y_train)

        joblib.dump(model, "random_forest_calories.pkl")
        return model

random_reg = train_or_load_model()

# Align prediction data columns with training data
df = df.reindex(columns=exercise_df.drop("Calories", axis=1).columns, fill_value=0)

# Predict calories burned
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
st.write(f"🔥 {round(prediction[0], 2)} **kilocalories burned**")

# Find similar results
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[
    (exercise_df["Calories"] >= calorie_range[0]) & 
    (exercise_df["Calories"] <= calorie_range[1])
]

# Health message based on prediction range
st.write("### Your Fitness Insight:")

if prediction[0] >= 150:
    st.success("💪 You're among the top 10% of highly active individuals in the dataset! Keep up the great work!")
elif prediction[0] >= 100:
    st.info("✅ You are in the healthy activity range, similar to about 70% of users. Great job maintaining a balanced routine!")
elif prediction[0] >= 50:
    st.warning("🟡 You're slightly below the average activity level. Consider increasing your exercise duration or intensity.")
else:
    st.error("🔴 You are in the lowest activity group. Try incorporating more physical activity into your routine for better health.")

st.write("---")
st.header("Similar Results: ")
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information: ")

# Compare user input with dataset
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write(f"You are older than {round(sum(boolean_age) / len(boolean_age), 2) * 100}% of users.")
st.write(f"Your exercise duration is higher than {round(sum(boolean_duration) / len(boolean_duration), 2) * 100}% of users.")
st.write(f"You have a higher heart rate than {round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}% of users.")
st.write(f"Your body temperature is higher than {round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}% of users.")
