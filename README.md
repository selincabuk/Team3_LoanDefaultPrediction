# Loan Default Prediction

This project aims to develop a machine learning model to predict loan defaults. By leveraging various borrower attributes and financial indicators, the model helps financial institutions make informed lending decisions and reduce the risk of defaults.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Model Development](#model-development)
- [Model Deployment and Prediction](#model-deployment-and-prediction)
- [Results and Future Work](#results-and-future-work)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The primary goal of this project is to create a robust model that can accurately predict whether a borrower will default on a loan. This prediction model is intended to support financial institutions in their decision-making processes, ultimately reducing the rate of loan defaults.

## Dataset
The dataset used for this project contains various borrower attributes such as yearly income, debt-to-income ratio, total unpaid credit limit, and more. The dataset is split into training and testing sets for model validation.

- **File Location**: `Dataset/Data_Train.csv`
- **Key Features**: `Yearly_Income`, `Debt_to_Income`, `Total_Unpaid_CL`, `Unpaid_Amount`, etc.

## Exploratory Data Analysis (EDA)
EDA is performed to understand the data distribution, relationships between variables, and to detect any anomalies or patterns. Key steps in EDA include:
- Distribution analysis of key variables.
- Visualization of pairwise relationships.
- Creation of a correlation matrix to identify multicollinearity.

## Preprocessing
Data preprocessing involves several steps to clean and transform the raw data into a format suitable for model training:
- **Reading Data**: Load the dataset from CSV file.
- **Imputation**: Fill missing values using KNN imputer for numerical columns and appropriate strategies for categorical columns.
- **Feature Engineering**: Create new features such as `Income_to_Debt_Ratio` and `Unpaid_Amount_to_Income_Ratio`.
- **Categorical Data Transformation**: Transform categorical variables using one-hot encoding or frequency encoding.
- **Cleaning and Transformation**: Clean and preprocess specific columns such as `Designation`, `GGGrade`, `Experience`, `Validation`, and `Home_Status`.

## Model Development
The model development process includes:
- **Model Selection**: Chose Random Forest Classifier for its robustness and ability to handle mixed data types.
- **Model Training**: Split the dataset into training and testing sets, train the model, and tune hyperparameters.
- **Model Evaluation**: Evaluate the model using metrics like accuracy, precision, recall, and F1 score. Generate a confusion matrix and classification report for detailed analysis.

## Model Deployment and Prediction
Developed a user interface using `tkinter` for users to input borrower data and receive default predictions:
- **User Interface**: Captures user inputs, processes them, and uses the trained model to predict loan default risk.
- **Prediction Results**: Demonstrates the prediction process with sample data.

## Results and Future Work
### Results
- Achieved an accuracy of [percentage] in predicting loan defaults, demonstrating the model's effectiveness.
- The prediction tool can aid financial institutions in reducing loan default rates by making informed lending decisions.

### Future Work
- **Enhancements**: Incorporate additional features and explore advanced machine learning techniques to improve accuracy.
- **Deployment**: Develop a web-based interface for broader accessibility and integrate the model into the lending decision process of financial institutions.

## Usage
To use this project:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-default-prediction.git
   cd loan-default-prediction
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
3. Run the preprocessing and model training script:
   ```bash
   python preprocessing.py
4. Launch the prediction interface:
 ```bash
    python predict.py

