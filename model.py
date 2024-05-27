import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import joblib
from preprocessing import read_data_file, fill_missing_values, feature_engineering, clean_designation, visualize_data, missing_values_info, process_features

def check_infinite_values(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def customer_segmentation(df, n_clusters=3):
    # Features for segmentation
    features = ['Yearly_Income', 'Debt_to_Income', 'Unpaid_2_years', 'Present_Balance']
    
    # Applying K-Means algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Segment'] = kmeans.fit_predict(df[features])
    
    return df

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

        print("\nProcessing features...")
        frame = process_features(frame)
        print("Features processed.")

        print("\nVisualizing data...")
        visualize_data(frame)
        print("Data visualization completed.")
        
        print("\nPerforming customer segmentation...")
        frame = customer_segmentation(frame)
        print("Customer segmentation completed.")
        print("Segment distribution:")
        print(frame['Segment'].value_counts())

        print("\nHandling categorical variables in new data...")
        categorical_columns = frame.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            num_unique = frame[col].nunique()
            print(f"Column '{col}' has {num_unique} unique values")
            if num_unique > 100:  # Threshold for high-cardinality columns
                print(f"Using frequency encoding for column '{col}'")
                # Frequency encoding
                freq_encoding = frame[col].value_counts().to_dict()
                frame[col] = frame[col].map(freq_encoding)
            else:
                print(f"Using one-hot encoding for column '{col}'")
                # One-hot encoding
                frame = pd.get_dummies(frame, columns=[col])

        print("\nChecking for infinite or very large values...")
        frame = check_infinite_values(frame)
        print("Check completed.")

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

        model_features = list(features.columns)
        joblib.dump(model_features, 'model_features.pkl')
        print("Model features saved as 'model_features.pkl'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
