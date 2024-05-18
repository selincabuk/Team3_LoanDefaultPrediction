import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from preprocessing import read_data_file, fill_missing_values, feature_engineering, clean_designation, visualize_data, missing_values_info

def check_infinite_values(df):
    # Very large values are detected and changed to NaN. These NaN values are then removed from the dataset.
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def main():
    try:
       
        frame = read_data_file('Dataset/Data_Train.csv')
        print("Data loaded successfully.")
        
       
        print("First five rows of the dataset:")
        print(frame.head())
        print("\nSize of the dataset:")
        print(frame.shape)
        print("\n")
        print(frame.describe())

       
        print("Missing values information:")
        missing_values_info(frame)

        
        print("\nFilling missing values...")
        frame = fill_missing_values(frame)
        print("Missing values filled.")

        print("\nPerforming feature engineering...")
        frame = feature_engineering(frame)
        print("Feature engineering completed.")

        print("\nCleaning designations...")
        frame = clean_designation(frame)
        print("Designations cleaned.")

        
        print("\nVisualizing data...")
        visualize_data(frame)
        print("Data visualization completed.")

        
        print("\nConverting categorical variables to numerical values...")
        frame = pd.get_dummies(frame)
        print("Conversion completed.")

        
        print("\nChecking for infinite or very large values...")
        frame = check_infinite_values(frame)
        print("Check completed.")

        #Separate target and features
        target = 'Default'  
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset.")
        
        features = frame.drop(columns=[target])
        labels = frame[target]

        print("\nSplitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        print("Data split completed.")

        print("\nTraining the model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model training completed.")

        print("\nMaking predictions on the test set...")
        y_pred = model.predict(X_test)
        print("Predictions completed.")

        
        print("\nEvaluating the model...")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        
        joblib.dump(model, 'loan_default_model.pkl')
        print("Model saved as 'loan_default_model.pkl'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
