# Student Grade Prediction

This project predicts student final grades (`G3`) using linear regression based on several features such as study time, past failures, and absences.

## Dataset
The `student-mat.csv` dataset contains the following features:
- `G1`: First period grade
- `G2`: Second period grade
- `G3`: Final grade (target variable)
- `studytime`: Time spent studying
- `failures`: Number of past failures
- `absences`: Number of absences

## Steps:
1. **Data Preprocessing**: Load the dataset, handle missing values, and prepare the features.
2. **Exploratory Data Analysis**: Analyze relationships between features and target variable (`G3`).
3. **Modeling**: Train a linear regression model to predict final grades.
4. **Evaluation**: Evaluate the model's performance with metrics like Mean Squared Error (MSE) and R².

## Running the Code:
1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn seaborn
