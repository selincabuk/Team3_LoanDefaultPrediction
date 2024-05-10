import pandas as pd

def count_null_values(df):
    # Her sütundaki null değerlerin sayısını hesapla
    null_counts = df.isnull().sum()

    # Her sütunun adını ve karşılık gelen null değer sayısını yazdır
    for column, null_count in null_counts.items():
        print(f"{column}: {null_count} null values")

def main():
    # Veri setini oku
    frame = pd.read_csv('Dataset/Data_Train.csv')

    # Null değerlerin sayısını belirle
    count_null_values(frame)

if __name__ == "__main__":
    main()
