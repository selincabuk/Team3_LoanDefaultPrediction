import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def read_data_file(dataset_file):
    df = pd.read_csv(dataset_file)
    return df

def impute_knn(df, predictor_cols, target_col, n_neighbors):
    # Subset of data containing non-missing values for predictor and target columns
    data_for_imputation = df.dropna(subset=predictor_cols + [target_col])

    x = data_for_imputation[predictor_cols]
    y = data_for_imputation[target_col]

    # Scale the predictor variables
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Initialize KNN imputer with specified number of neighbors
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Fit the imputer on the scaled data
    imputer.fit(x_scaled)

    # Extract rows with missing target
    missing_data = df[df[target_col].isna()]

    # Scale the missing data using the same scaler
    missing_data_scaled = scaler.transform(missing_data[predictor_cols])

    # Impute target
    imputed_values = imputer.transform(missing_data_scaled)

    # Assign missing target values
    df.loc[df[target_col].isna(), target_col] = imputed_values[:, 0]

    return df

def data_preprocessing(df):
    # Checking the number of missing values
    print("Number of missing values:")
    print(df.isnull().sum())
    print("\n")

    # Attribute minimum and maximum values
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            max_value = np.max(pd.to_numeric(df[column], errors='coerce'))
            min_value = np.min(pd.to_numeric(df[column], errors='coerce'))
            # Print attribute name, maximum value, and minimum value
            print(f"Attribute: {column}, Maximum: {max_value}, Minimum: {min_value}")

    # Fill in null values
    # Designation will be N/A if null
    df['Designation'] = df['Designation'].fillna('N/A')
    # Postal Code will be chosen randomly between 99900.0 - 1000.0
    df['Postal_Code'] = df['Postal_Code'].fillna(np.random.randint(1000, 99900 + 1))
    # Debt_to_Income, Total_Unpaid_CL, Unpaid_Amount, Yearly_Income fill by imputation

    # Plot Debt_to_Income vs Yearly_Income
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Yearly_Income'], df['Debt_to_Income'])
    plt.title('Debt-to-Income Ratio vs Yearly Income')
    plt.xlabel('Yearly Income ($)')
    plt.ylabel('Debt-to-Income Ratio (%)')
    plt.savefig('Debt_to_Income_vs_Yearly_Income.png')
    plt.close()

    # Plot Experience vs Yearly_Income
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Yearly_Income'], df['Experience'])
    plt.title('Debt-to-Income Ratio vs Yearly Income')
    plt.xlabel('Yearly Income ($)')
    plt.ylabel('Experience (yrs)')
    plt.savefig('Experience_vs_Yearly_Income.png')
    plt.close()

    # Plot Total_Unpaid_CL vs Yearly_Income
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Yearly_Income'], df['Total_Unpaid_CL'])
    plt.title('Debt-to-Income Ratio vs Yearly Income')
    plt.xlabel('Yearly Income ($)')
    plt.ylabel('Total_Unpaid_CL')
    plt.savefig('Total_Unpaid_CL_vs_Yearly_Income.png')
    plt.close()

    # Plot Unpaid_Amount vs Yearly_Income
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Yearly_Income'], df['Unpaid_Amount'])
    plt.title('Unpaid_Amount vs Yearly Income')
    plt.xlabel('Yearly Income ($)')
    plt.ylabel('Unpaid_Amount')
    plt.savefig('Unpaid_Amount.png')
    plt.close()

    # Plot Asst_Reg vs Yearly_Income
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Yearly_Income'], df['Asst_Reg'])
    plt.title('Asst_Reg vs Yearly Income')
    plt.xlabel('Yearly Income ($)')
    plt.ylabel('Asst_Reg')
    plt.savefig('Asst_Reg.png')
    plt.close()

    # There is no clear correlation between potential predictors and yearly income, also no interaction terms
    # Use KNN to impute yearly income
    predictor_cols = ['Debt_to_Income', 'Total_Unpaid_CL', 'Unpaid_Amount']
    target_col = 'Yearly_Income'
    impute_knn(df, predictor_cols, target_col, n_neighbors=10)

    # Use KNN to impute Debt_to_Income
    predictor_cols = ['Yearly_Income', 'Total_Unpaid_CL', 'Unpaid_Amount']
    target_col = 'Debt_to_Income'
    impute_knn(df, predictor_cols, target_col, n_neighbors=10)

    # Use KNN to impute yearly income
    predictor_cols = ['Yearly_Income', 'Debt_to_Income', 'Unpaid_Amount']
    target_col = 'Total_Unpaid_CL'
    impute_knn(df, predictor_cols, target_col, n_neighbors=10)

    # Use KNN to impute yearly income
    predictor_cols = ['Yearly_Income', 'Debt_to_Income', 'Total_Unpaid_CL']
    target_col = 'Unpaid_Amount'
    impute_knn(df, predictor_cols, target_col, n_neighbors=10)

    
    # Lend amount and usage rate scatter plot
    sns.scatterplot(x='Lend_Amount', y='Usage_Rate', data=df)
    plt.title('Usage Rate vs. Lend Amount')
    plt.xlabel('Lend Amount')
    plt.ylabel('Usage Rate')
    plt.savefig('usage_rate_vs_lend_amount.png')
    plt.close()
    # Assign mean value to usage rate outlier
    max_usage_rate_row = df['Usage_Rate'].idxmax()
    mean_usage_rate = df['Usage_Rate'].mean()
    df.loc[max_usage_rate_row, 'Usage_Rate'] = mean_usage_rate

    # Huge amount of distinct designations
    """for designation in df['Designation'].unique():
        print(designation)"""
    
    # Count the occurrences of each designation
    designation_counts = df['Designation'].value_counts()
    #print(designation_counts[:20])

    # Group rare designations into Other
    rare_designations = designation_counts[designation_counts < 2].index
    df['Designation'] = df['Designation'].apply(lambda x: 'Other' if x in rare_designations else x)

    # We have 34k as Other and the closest to this is 1500 with school teacher, not sure how to handle this
    #print(df['Designation'].value_counts().get('Other', 0))

    # Plot account openings
    # There is one with 83 openings, i guess it's possible
    plt.figure(figsize=(10, 6))
    plt.scatter(df['ID'], df['Account_Open'])
    plt.title('Account Openings vs. ID')
    plt.xlabel('ID')
    plt.ylabel('Number of Account Openings')
    plt.savefig('account_openings.png')
    plt.close()


def missing_values_info(data):
    for column in data.columns:
        missing = data[column].isna().sum()
        portion = (missing / data.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")



    

    
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
    print(frame.describe)
    # Applying data preprocessing steps
    cleaned_data = data_preprocessing(frame)
    
    

if __name__ == "__main__":
    main()
