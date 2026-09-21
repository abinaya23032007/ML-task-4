# Polynomial Regression for Household Energy Consumption

## 📌 Project Overview

This project uses **Polynomial Regression** to predict household energy consumption based on household and usage-related features.

The model uses:

* Household Size
* Average Temperature
* Peak Hours Usage

The target variable is **Energy Consumption**.

---

## 🎯 Objective

The main objective of this project is to build a machine learning model that can predict household energy consumption and evaluate its performance using different regression metrics.

---

## 📂 Dataset

The program uses the following dataset:

`household_energy_consumption.csv`

### Input Features

* `Household_Size` – Number of people in the household.
* `Avg_Temperature_C` – Average temperature in Celsius.
* `Peak_Hours_Usage_kWh` – Energy usage during peak hours.

### Target Variable

* `Energy_Consumption_kWh` – Total household energy consumption.

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab / Jupyter Notebook

---

## 📦 Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## 🔄 Project Workflow

### 1. Load the Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("/content/household_energy_consumption.csv")
```

### 2. Explore the Dataset

The program checks:

* First few records
* Dataset information
* Statistical summary
* Dataset shape
* Column names
* Missing values

### 3. Select Features and Target

The following features are selected:

```python
x = df[["Household_Size",
        "Avg_Temperature_C",
        "Peak_Hours_Usage_kWh"]]

y = df["Energy_Consumption_kWh"]
```

### 4. Split the Dataset

The data is divided into training and testing sets.

* **80%** → Training data
* **20%** → Testing data

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

### 5. Create Polynomial Features

Polynomial features are created with **degree 2**.

```python
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

This allows the model to capture non-linear relationships between the input features and energy consumption.

### 6. Train the Model

A Linear Regression model is trained using the polynomial features.

```python
model = LinearRegression()
model.fit(X_train_poly, Y_train)
```

### 7. Make Predictions

The trained model predicts energy consumption for the test data.

```python
Y_pred = model.predict(X_test_poly)
```

---

## 📊 Model Evaluation

The model is evaluated using four metrics:

### MAE – Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

### MSE – Mean Squared Error

Measures the average squared difference between actual and predicted values.

### RMSE – Root Mean Squared Error

It is the square root of MSE and represents prediction error in the same unit as the target.

### R² Score

Shows how well the model explains the variation in energy consumption.

The program calculates these metrics:

```python
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_test, Y_pred)
```

The evaluation section is implemented directly in the uploaded program.

---

## 📈 Visualization

The project creates a scatter plot comparing:

* **Actual Energy Consumption**
* **Predicted Energy Consumption**

```python
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual Energy Consumption (KWh)")
plt.ylabel("Predicted Energy Consumption (KWh)")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()
```

This visualization helps understand how closely the predicted values match the actual values.

---

## ▶️ How to Run

### Using Google Colab

1. Open Google Colab.
2. Upload the Python file/notebook.
3. Upload `household_energy_consumption.csv`.
4. Make sure the CSV path matches the path used in the program.
5. Run the cells in order.
6. Check the model evaluation metrics and graph.

### Using Jupyter Notebook

1. Install the required libraries.
2. Place the CSV dataset in the appropriate folder.
3. Open `polynomial.ipynb`.
4. Run all cells.

Install required packages using:

```bash
pip install pandas matplotlib scikit-learn
```

---

## 📁 Project Structure

```text
Polynomial-Regression/
│
├── polynomial.py
├── polynomial.ipynb
├── household_energy_consumption.csv
└── README.md
```

---

## ✅ Output

The program produces:

1. Dataset information and statistical details.
2. Model evaluation results:

   * MAE
   * MSE
   * RMSE
   * R² Score
3. An **Actual vs Predicted Energy Consumption** scatter plot.

---

## 📌 Conclusion

This project demonstrates how **Polynomial Regression** can be used for household energy consumption prediction. Polynomial features allow the regression model to represent non-linear relationships between household characteristics, temperature, peak-hour usage, and energy consumption.
