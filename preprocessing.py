import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import re


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
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
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


def transform_GGGrade(df):
    roman_to_int = {
    'I': 1,
    'II': 2,
    'III': 3,
    'IV': 4,
    'V': 5,
    'VI': 6
    }
    df['GGGrade'] = df['GGGrade'].map(roman_to_int)

    return df


def preprocess_experience_helper(value):
    # Handle '>' and '<' cases
    if '>' in value:
        return int(re.findall(r'\d+', value)[0]) + 1
    elif '<' in value:
        return 0  # Assuming '<1yr' means less than 1 year, we can consider it as 0
    else:
        # Extract numeric part from the string
        nums = re.findall(r'\d+', value)
        if nums:
            return int(nums[0])
        else:
            return 0  # In case there's no numeric part, default to 0


def transform_Experience(df):
    df['Experience'] = df['Experience'].apply(preprocess_experience_helper)

    return df


def transform_Validation(df):
    status_mapping = {
    'Vfied': 1,
    'Source Verified': 1, # these are both verified so they get the same value
    'Not Vfied': 0
    }

    df['Validation'] = df['Validation'].map(status_mapping)

    return df


def transform_Home_Status(df):
    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=['Home_Status'], prefix='', prefix_sep='')

    return df

def remove_Designation(df):
    df = df.drop(columns=['Designation'])

    return df

def remove_File_Status(df):
    df = df.drop(columns=['File_Status'])

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

def process_features(frame):
    frame = transform_GGGrade(frame)
    frame = transform_Experience(frame)
    frame = transform_Validation(frame)
    frame = transform_Home_Status(frame)
    frame = remove_Designation(frame)
    frame = remove_File_Status(frame)

    return frame
        

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

    # Select non-numeric columns
    non_numeric_columns = frame.select_dtypes(exclude=['number'])

    # Print non-numeric columns
    print("Non-numeric columns:")
    print(non_numeric_columns)

    # Print the names of the non-numeric columns
    print("\nNames of non-numeric columns:")
    print(non_numeric_columns.columns.tolist())

    print("Experience column unique values:")
    print(frame['Experience'].unique())

    print("Validation column unique values:")
    print(frame['Validation'].unique())

    print("Home status column unique values:")
    print(frame['Home_Status'].unique())

    print("Designation column unique values:")
    print(frame['Designation'].unique().size)

    frame = process_features(frame)

    #save clean data
    output_file = 'Dataset/Clean_data.csv'
    frame.to_csv(output_file, index=False)
    print(f"Clean data saved to '{output_file}'")

    visualize_data(frame)


if __name__ == "__main__":
    main()
