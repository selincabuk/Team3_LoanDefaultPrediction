import pandas as pd
import joblib
from preprocessing import read_data_file, fill_missing_values, feature_engineering, clean_designation, check_infinite_values

def main():
    try:
        # Yeni veri dosyasını oku
        new_data_file = 'Dataset/New_Data.csv'  # Tahmin yapmak istediğiniz yeni veri dosyası olmalı
        frame = read_data_file(new_data_file)
        print("New data loaded successfully.")


        # Kayıp değerleri doldurma
        print("\nFilling missing values...")
        frame = fill_missing_values(frame)
        print("Missing values filled.")

        # Özellik mühendisliği
        print("\nPerforming feature engineering...")
        frame = feature_engineering(frame)
        print("Feature engineering completed.")

        # Designation sütununu temizleme
        print("\nCleaning designations...")
        frame = clean_designation(frame)
        print("Designations cleaned.")

        # Kategorik değişkenleri sayısal değerlere çevirme
        print("\nConverting categorical variables to numerical values...")
        frame = pd.get_dummies(frame)
        print("Conversion completed.")

        # Sonsuz veya çok büyük değerleri kontrol etme
        print("\nChecking for infinite or very large values...")
        frame = check_infinite_values(frame)
        print("Check completed.")

        # Modeli yükleme
        model = joblib.load('loan_default_model.pkl')
        print("Model loaded successfully.")

        # Yeni veri kümesi ile eğitim veri kümesi arasındaki özellik uyumsuzluklarını giderme
        model_features = model.feature_names_in_
        for feature in model_features:
            if feature not in frame.columns:
                frame[feature] = 0

        frame = frame[model_features]

        # Tahmin yapma
        predictions = model.predict(frame)
        frame['Default_Prediction'] = predictions

        # Sonuçları kaydetme
        frame.to_csv('Dataset/New_Data_Predictions.csv', index=False)
        print("Predictions saved to 'Dataset/New_Data_Predictions.csv'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
