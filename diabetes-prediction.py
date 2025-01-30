from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os

# Load the dataset and prepare the model
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=column_names)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train different models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)

app = Flask(__name__)

# Ensure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    model_used = None
    plot_url = None
    prediction_class = None

    if request.method == 'POST':
        try:
            # Get user inputs from form
            pregnancies = int(request.form['Pregnancies'])
            glucose = int(request.form['Glucose'])
            blood_pressure = int(request.form['BloodPressure'])
            skin_thickness = int(request.form['SkinThickness'])
            insulin = int(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
            age = int(request.form['Age'])

            user_data = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree_function, age
            ]])

            # Standardize user data
            user_data_scaled = scaler.transform(user_data)

            # Make predictions using all models
            rf_prediction = rf_model.predict(user_data_scaled)[0]
            svm_prediction = svm_model.predict(user_data_scaled)[0]
            lr_prediction = lr_model.predict(user_data_scaled)[0]
            ann_prediction = ann_model.predict(user_data_scaled)[0]

            # Store predictions
            predictions = {
                "Random Forest": "Diabetic" if rf_prediction == 1 else "Non-Diabetic",
                "SVM": "Diabetic" if svm_prediction == 1 else "Non-Diabetic",
                "Logistic Regression": "Diabetic" if lr_prediction == 1 else "Non-Diabetic",
                "ANN": "Diabetic" if ann_prediction == 1 else "Non-Diabetic",
            }

            # Determine the overall prediction (Majority Vote)
            from collections import Counter
            model_predictions = list(predictions.values())
            most_common_prediction = Counter(model_predictions).most_common(1)[0][0]
            prediction = most_common_prediction
            model_used = "Ensemble (Majority Vote)"

            # Determine prediction class for styling:
            if prediction == "Diabetic":
                prediction_class = "error"
            elif prediction == "Non-Diabetic":
                prediction_class = "success"
            else:
                prediction_class = ""

            # Plot the predictions of each model
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(predictions.keys()), y=[1 if val == "Diabetic" else 0 for val in predictions.values()], palette="coolwarm")
            plt.title("Model Predictions")
            plt.xlabel("Model")
            plt.ylabel("Prediction (1 = Diabetic, 0 = Non-Diabetic)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('static/prediction_plot.png')

            plot_url = 'static/prediction_plot.png'

        except Exception as e:
            prediction = f"Error: {str(e)}"
            prediction_class = "error"

    return render_template('index.html', prediction=prediction, model_used=model_used, plot_url=plot_url, prediction_class=prediction_class)

if __name__ == '__main__':
    app.run(debug=True)