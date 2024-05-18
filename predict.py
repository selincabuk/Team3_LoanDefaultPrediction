import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import plotly.express as px

# Load the saved model
model = load('loan_default_model.pkl')

def impute_knn(df, predictor_cols, target_col, n_neighbors=10):
    data_for_imputation = df.dropna(subset=predictor_cols + [target_col])

    x = data_for_imputation[predictor_cols]
    y = data_for_imputation[target_col]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(x_scaled)

    missing_data = df[df[target_col].isna()]
    missing_data_scaled = scaler.transform(missing_data[predictor_cols])

    imputed_values = imputer.transform(missing_data_scaled)
    df.loc[df[target_col].isna(), target_col] = imputed_values[:, 0]

    return df
# Function to preprocess new data (similar structure to preprocessing.py)
def feature_engineering(df):
    # Create new features
    df['Income_to_Debt_Ratio'] = df['Yearly_Income'] / df['Debt_to_Income']
    df['Unpaid_Amount_to_Income_Ratio'] = df['Unpaid_Amount'] / df['Yearly_Income']

    return df


def clean_designation(df):
    designation_counts = df['Designation'].value_counts()
    rare_designations = designation_counts[designation_counts < 2].index
    df['Designation'] = df['Designation'].apply(lambda x: 'Other' if x in rare_designations else x)

    return df


def fill_missing_values(df):
    df['Designation'] = df['Designation'].fillna('N/A')
    df['Postal_Code'] = df['Postal_Code'].fillna(np.random.randint(1000, 99900 + 1))

    predictor_target_pairs = [
        (['Debt_to_Income', 'Total_Unpaid_CL', 'Unpaid_Amount'], 'Yearly_Income'),
        (['Yearly_Income', 'Total_Unpaid_CL', 'Unpaid_Amount'], 'Debt_to_Income'),
        (['Yearly_Income', 'Debt_to_Income', 'Unpaid_Amount'], 'Total_Unpaid_CL'),
        (['Yearly_Income', 'Debt_to_Income', 'Total_Unpaid_CL'], 'Unpaid_Amount')
    ]

    for predictors, target in predictor_target_pairs:
        df = impute_knn(df, predictors, target)

    return df


def main():
  # Load your new data
  new_data = pd.read_csv('Dataset\Data_Train.csv')

  # Preprocess the new data
  
  preprocessed_data = fill_missing_values(new_data)
  preprocessed_data = feature_engineering(preprocessed_data)
  preprocessed_data = clean_designation(preprocessed_data)


  # Make predictions
  predictions = model.predict(preprocessed_data)

  # Print the predictions (you can modify this to suit your needs)
  print("Predicted Loan Defaults:")
  print(predictions)

if __name__ == "__main__":
  main()
