import tensorflow
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import pickle
import seaborn as sns
import sklearn
from matplotlib import style
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Loading and Preparing the data
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Load the best model after training
with open("student_model.pickle", "rb") as pickle_in:
    linear = pickle.load(pickle_in)

# Print the coefficients and intercept
print('Coefficient:\n', linear.coef_)
print('Intercept: \n', linear.intercept_)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Make predictions using the best model
predictions = linear.predict(x_test)

# Print predictions along with features and actual values
for i in range(len(predictions)):
    print(f"Prediction: {predictions[i]}, Features: {x_test[i]}, Actual: {y_test[i]}")

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R²: {r2}")

# Basic scatter plot
style.use("ggplot")
pyplot.scatter(data["absences"], data["G3"])
pyplot.xlabel("Absences")
pyplot.ylabel("Final Grade")
pyplot.savefig("absences_vs_final_grade.png", dpi=300, bbox_inches='tight')
pyplot.show()

# Actual vs predicted scatter plot
pyplot.scatter(y_test, predictions)
pyplot.xlabel('Actual Grades')
pyplot.ylabel('Predicted Grades')
pyplot.title('Actual vs Predicted Grades')
pyplot.show()

# Seaborn-style plot
sns.set(style="whitegrid")

# Create the scatter plot
pyplot.figure(figsize=(8, 6))
scatter = pyplot.scatter(data["absences"], data["G3"], c=data["G3"], cmap='coolwarm', alpha=0.7, edgecolors='w', s=80)

# Title and labels
pyplot.title('Relationship between Absences and Final Grade', fontsize=16, fontweight='bold')
pyplot.xlabel("Number of Absences", fontsize=12)
pyplot.ylabel("Final Grade (G3)", fontsize=12)

# Color bar
cbar = pyplot.colorbar(scatter)
cbar.set_label('Final Grade (G3)', fontsize=12)

x
for i in range(0, len(data), 10):
    pyplot.text(data["absences"].iloc[i], data["G3"].iloc[i], str(i), fontsize=8, color='black')


pyplot.grid(True, linestyle='--', alpha=0.7)
pyplot.tight_layout()
pyplot.show()
