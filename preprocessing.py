import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px


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


def interactive_plot(df, x_col, y_col, title, xlabel, ylabel, filename):
    fig = px.scatter(df, x=x_col, y=y_col, title=title, labels={x_col: xlabel, y_col: ylabel})
    fig.write_html(filename)


def visualize_data(df):
    plot_scatter(df, 'Yearly_Income', 'Debt_to_Income', 'Debt-to-Income Ratio vs Yearly Income', 'Yearly Income ($)', 'Debt-to-Income Ratio (%)', 'Debt_to_Income_vs_Yearly_Income.png')
    interactive_plot(df, 'Yearly_Income', 'Experience', 'Experience vs Yearly Income', 'Yearly Income ($)', 'Experience (yrs)', 'Experience_vs_Yearly_Income.html')
    interactive_plot(df, 'Yearly_Income', 'Total_Unpaid_CL', 'Total Unpaid Credit Limit vs Yearly Income', 'Yearly Income ($)', 'Total Unpaid CL', 'Total_Unpaid_CL_vs_Yearly_Income.html')
    interactive_plot(df, 'Yearly_Income', 'Unpaid_Amount', 'Unpaid Amount vs Yearly Income', 'Yearly Income ($)', 'Unpaid Amount', 'Unpaid_Amount_vs_Yearly_Income.html')
    interactive_plot(df, 'Yearly_Income', 'Asst_Reg', 'Asst Reg vs Yearly Income', 'Yearly Income ($)', 'Asst Reg', 'Asst_Reg_vs_Yearly_Income.html')

    sns.scatterplot(x='Lend_Amount', y='Usage_Rate', data=df)
    plt.title('Usage Rate vs. Lend Amount')
    plt.xlabel('Lend Amount')
    plt.ylabel('Usage Rate')
    plt.savefig('usage_rate_vs_lend_amount.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['ID'], df['Account_Open'])
    plt.title('Account Openings vs. ID')
    plt.xlabel('ID')
    plt.ylabel('Number of Account Openings')
    plt.savefig('account_openings.png')
    plt.close()

    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Missing Data Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Data Heatmap')
    plt.savefig('missing_data_heatmap.png')
    plt.close()


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

    missing_values_info(frame)
    
    frame = fill_missing_values(frame)
    frame = feature_engineering(frame)
    frame = clean_designation(frame)

    visualize_data(frame)


if __name__ == "__main__":
    main()
