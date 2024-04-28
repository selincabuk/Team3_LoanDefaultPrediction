import pandas as pd
import numpy as np
import seaborn as sns

def read_data_file(dataset_file):
    df = pd.read_csv(dataset_file)
    return df

def data_preprocessing(df):
    # Checking the number of missing values
    print("Number of missing values:")
    print(df.isnull().sum())
    print("\n")
    
def main():
    # Reading the dataset file
    frame = read_data_file('Dataset/Data_Train.csv')

    # Displaying the first five rows of the dataset
    print("First five rows of the dataset:")
    print(frame.head())

    # Displaying the size of the dataset
    print("\nSize of the dataset:")
    print(frame.shape)
    print("\n")
    
    # Applying data preprocessing steps
    cleaned_data = data_preprocessing(frame)

if __name__ == "__main__":
    main()
