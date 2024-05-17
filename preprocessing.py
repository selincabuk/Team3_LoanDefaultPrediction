import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def read_data_file(dataset_file):
    df = pd.read_csv(dataset_file)
    return df


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


def plot_scatter(df, x_col, y_col, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


def data_preprocessing(df):
    print("Number of missing values:")
    print(df.isnull().sum())
    print("\n")

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            max_value = np.max(pd.to_numeric(df[column], errors='coerce'))
            min_value = np.min(pd.to_numeric(df[column], errors='coerce'))
            print(f"Attribute: {column}, Maximum: {max_value}, Minimum: {min_value}")

    df['Designation'] = df['Designation'].fillna('N/A')
    df['Postal_Code'] = df['Postal_Code'].fillna(np.random.randint(1000, 99900 + 1))

    plot_scatter(df, 'Yearly_Income', 'Debt_to_Income', 'Debt-to-Income Ratio vs Yearly Income', 'Yearly Income ($)', 'Debt-to-Income Ratio (%)', 'Debt_to_Income_vs_Yearly_Income.png')
    plot_scatter(df, 'Yearly_Income', 'Experience', 'Experience vs Yearly Income', 'Yearly Income ($)', 'Experience (yrs)', 'Experience_vs_Yearly_Income.png')
    plot_scatter(df, 'Yearly_Income', 'Total_Unpaid_CL', 'Total Unpaid Credit Limit vs Yearly Income', 'Yearly Income ($)', 'Total Unpaid CL', 'Total_Unpaid_CL_vs_Yearly_Income.png')
    plot_scatter(df, 'Yearly_Income', 'Unpaid_Amount', 'Unpaid Amount vs Yearly Income', 'Yearly Income ($)', 'Unpaid Amount', 'Unpaid_Amount_vs_Yearly_Income.png')
    plot_scatter(df, 'Yearly_Income', 'Asst_Reg', 'Asst Reg vs Yearly Income', 'Yearly Income ($)', 'Asst Reg', 'Asst_Reg_vs_Yearly_Income.png')

    predictor_target_pairs = [
        (['Debt_to_Income', 'Total_Unpaid_CL', 'Unpaid_Amount'], 'Yearly_Income'),
        (['Yearly_Income', 'Total_Unpaid_CL', 'Unpaid_Amount'], 'Debt_to_Income'),
        (['Yearly_Income', 'Debt_to_Income', 'Unpaid_Amount'], 'Total_Unpaid_CL'),
        (['Yearly_Income', 'Debt_to_Income', 'Total_Unpaid_CL'], 'Unpaid_Amount')
    ]

    for predictors, target in predictor_target_pairs:
        impute_knn(df, predictors, target)

    sns.scatterplot(x='Lend_Amount', y='Usage_Rate', data=df)
    plt.title('Usage Rate vs. Lend Amount')
    plt.xlabel('Lend Amount')
    plt.ylabel('Usage Rate')
    plt.savefig('usage_rate_vs_lend_amount.png')
    plt.close()

    max_usage_rate_row = df['Usage_Rate'].idxmax()
    mean_usage_rate = df['Usage_Rate'].mean()
    df.loc[max_usage_rate_row, 'Usage_Rate'] = mean_usage_rate

    designation_counts = df['Designation'].value_counts()
    rare_designations = designation_counts[designation_counts < 2].index
    df['Designation'] = df['Designation'].apply(lambda x: 'Other' if x in rare_designations else x)

    plt.figure(figsize=(10, 6))
    plt.scatter(df['ID'], df['Account_Open'])
    plt.title('Account Openings vs. ID')
    plt.xlabel('ID')
    plt.ylabel('Number of Account Openings')
    plt.savefig('account_openings.png')
    plt.close()

    return df


def missing_values_info(data):
    for column in data.columns:
        missing = data[column].isna().sum()
        portion = (missing / data.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")


def main():
    frame = read_data_file('Dataset/Data_Train.csv')

    print("First five rows of the dataset:")
    print(frame.head())

    print("\nSize of the dataset:")
    print(frame.shape)
    print("\n")
    print(frame.describe())

    cleaned_data = data_preprocessing(frame)


if __name__ == "__main__":
    main()
