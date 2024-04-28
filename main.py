import pandas as pd
import numpy as np
import seaborn as sns

def read_the_data_file(dataset_file):
    df = pd.read_csv(dataset_file)
    return df


def main():
    
    frame = read_the_data_file('Dataset/Data_Train.csv')

    print("The first 5 row of the dataset:")
    print(frame.head())

    
    print("\nThe size of the dataset:")
    print(frame.shape)

if __name__ == "__main__":
    main()
