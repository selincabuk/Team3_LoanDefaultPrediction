import pandas as pd
import numpy as np
import joblib
from preprocessing import read_data_file, fill_missing_values, feature_engineering, clean_designation
from model import check_infinite_values

def main():
    try:
        model = joblib.load('loan_default_model.pkl')
        print("Model loaded successfully.")
        
        new_data_file = 'Dataset/Data_Train.csv'
        new_data = read_data_file(new_data_file)
        print("New data loaded successfully.")
        
        print("\nFirst five rows of the new dataset:")
        print(new_data.head())
        print("\nSize of the new dataset:")
        print(new_data.shape)
        print("\n")
        print(new_data.describe())

        print("\nFilling missing values in new data...")
        new_data = fill_missing_values(new_data)
        print("Missing values filled.")

        print("\nPerforming feature engineering on new data...")
        new_data = feature_engineering(new_data)
        print("Feature engineering completed.")

        print("\nCleaning designations in new data...")
        new_data = clean_designation(new_data)
        print("Designations cleaned.")

        print("\nConverting categorical variables to numerical values in new data...")
        new_data = pd.get_dummies(new_data)
        print("Conversion completed.")

        print("\nChecking for infinite or very large values in new data...")
        new_data = check_infinite_values(new_data)
        print("Check completed.")

        model_features = joblib.load('model_features.pkl')
        missing_cols = set(model_features) - set(new_data.columns)
        for col in missing_cols:
            new_data[col] = 0
        new_data = new_data[model_features]

        print("\nMaking predictions on the new data...")
        predictions = model.predict(new_data)
        new_data['Default_Prediction'] = predictions
        print("Predictions completed.")

        output_file = 'Dataset/Predictions.csv'
        new_data.to_csv(output_file, index=False)
        print(f"Predictions saved to '{output_file}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
