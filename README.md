🎮 Video Game Sales Prediction

This project uses machine learning to predict whether a video game will achieve good or bad global sales based on features such as platform, genre, and publisher.

🧠 Project Overview

Exploratory Data Analysis (EDA) – Statistical analysis and visualization of the dataset to understand sales distribution, detect outliers, and identify patterns between features.

Data Preprocessing – Cleaning the dataset, handling missing values, and removing outliers.

AutoGluon Modeling – Automated model selection to identify the best-performing algorithm.

Feature Engineering – Creation of statistical, interaction, and ranking features to improve accuracy.

Model Evaluation – Best F1-test score of 0.74 achieved using LightGBM.

Manual Retraining – The selected LightGBM model was retrained separately to optimize memory usage.

Deployment – The trained model was integrated into a Streamlit app for real-time predictions.

⚙️ Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn

AutoGluon, LightGBM, Scikit-learn

Streamlit

🌍 Live Demo

Try the hosted version here:

https://game-sales-predictor-wuzfyg3z7uhrynk2tas3pb.streamlit.app/

📈 Results

Accuracy: 73% (with feature engineering)

Macro F1-score: 0.73

Best model: LightGBM (F1-test = 0.74)

📦 Model File

Trained model: lightgbm_sales_classifier.pkl
