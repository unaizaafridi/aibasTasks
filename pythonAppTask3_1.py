import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic


def load_clean_dataset(df):
    
    # Data cleaning
    # df_cleaned = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    # df_cleaned = df_cleaned.dropna()  # Drop NaN values
    
     # Ensure all columns are numeric, coercing errors
    df_cleaned = df.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    df_cleaned = df_cleaned.dropna()

    return df_cleaned

def drop_outliers_iqr(data, threshold=1.5):
    # Create a mask to keep rows where not all columns are outliers
    mask = np.ones(len(data), dtype=bool)  # Start with all True

    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identify outliers for the current column
        outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
        
        # Update the mask to keep rows where not all columns are outliers
        mask &= ~outliers  # Keep rows that are not outliers in the current column

    # Return the data where the mask is False (i.e., not all columns are outliers)
    return data[~mask]


def drop_outliers_zscore(data, threshold=3):
    
    # Replace infinite values with NaN and drop them
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # Apply log transformation to scale down extreme values
    data = data.apply(lambda x: np.log1p(x) if (x > 0).all() else x)
    
    z_scores = np.abs(stats.zscore(data))  # Z-score for all columns
    data = data[(z_scores < threshold).all(axis=1)]  # Keep rows where all columns are below the threshold

    return data

def normalize_data(data):
    # Normalize the data using Min-Max normalization
    return (data - data.min()) / (data.max() - data.min())



def main():
    # Load the dataset (replace with the actual path to dataset02.csv)
    df = pd.read_csv('dataset02.csv')
    
    # Print original data
    print("Original Data:")
    print(df.head())
    print(f"Initial data shape: {df.shape}")
    
    # Step 1: Print the data types of the columns
    print("\nData Types:")
    print(df.dtypes)
    
    # Optional: If non-numeric values exist, convert them to numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    data = load_clean_dataset(df)
    print("\nload_clean_dataset:")
    print(data.head())
    print(f"Initial data shape: {data.shape}")

     # Drop outliers using Z-score filter
    data = drop_outliers_zscore(data)
    print("\ndrop_outliers_zscore:")
    print(data.head())
    print(f"Initial data shape: {data.shape}")

    # Drop outliers using IQR filter
    # data = drop_outliers_iqr(data)
    # print("\ndrop_outliers_iqr:")
    # print(data.head())
    # print(f"Initial data shape: {data.shape}")

    # Normalize the data
    data = normalize_data(data)

    # Print the cleaned and normalized data
    print("\normalize_data:")
    print(data.head())
    print(f"Initial data shape: {data.shape}")

     # Optionally, plot data
    if not data.empty:
        print("Plotting data...")  # Debugging statement
        data.plot(kind='line')
        plt.title("Normalized Data")
        plt.xlabel("Index")
        plt.ylabel("Normalized Values")
        plt.show()
        # plt.savefig('plot.png')
    else:
        print("No data left to plot.")
        
    # Split the dataset into training and testing sets (80% training, 20% testing)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    print("Columns in training data:", train_data.columns)

    if 'x' not in train_data.columns or 'y' not in train_data.columns:
        print("Error: 'x' or 'y' column missing in train_data")
        return
    
    # Save the datasets to CSV files
    train_data.to_csv('dataset02_training.csv', index=False)
    test_data.to_csv('dataset02_testing.csv', index=False)
    
    # Fit the OLS model using the training data
    X_train = train_data.drop('y', axis=1)  # Replace 'target_variable' with the actual target column name
    y_train = train_data['y']
    X_train = sm.add_constant(X_train)  # Adds a constant term to the predictor

    ols_model = sm.OLS(y_train, X_train).fit()
    
    
    # Visualization: Scatter Plot
    plt.figure(figsize=(12, 6))

    # Scatter plot for training data
    plt.scatter(train_data['x'], train_data['y'], color='orange', label='Training Data')  
    # Scatter plot for testing data
    plt.scatter(test_data['x'], test_data['y'], color='blue', label='Testing Data')  

    # Red line plot for OLS predictions
    x_values = np.linspace(train_data['x'].min(), train_data['x'].max(), 100)
    y_values = ols_model.predict(sm.add_constant(x_values))
    plt.plot(x_values, y_values, color='red', label='OLS Fit')
    
    plt.title('Scatter Plot of Influence Data vs. Target Variable')
    plt.xlabel('Influence Data')
    plt.ylabel('Target Variable')
    plt.legend()
    plt.savefig('UE_04_App2_ScatterVisualizationAndOlsModel.pdf')
    plt.close()
    
    # Box plot
    plt.figure(figsize=(12, 6))
    train_data.boxplot()
    plt.title('Box Plot of All Dimensions of Training Data')
    plt.savefig('UE_04_App2_BoxPlot.pdf')
    plt.close()

    # Diagnostic Plots using the LinearRegDiagnostic class
    # diagnostic = LinearRegDiagnostic(ols_model)
    # diagnostic(plt.style.context('seaborn-v0_8-paper'))
    # plt.savefig('UE_04_App2_DiagnosticPlots.pdf')
    # plt.close()


if __name__ == '__main__':
    main()