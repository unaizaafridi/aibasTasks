import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('dataset02.csv')

# Check data types
print(data.dtypes)

# Convert columns to numeric, forcing errors to NaN
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')

# Drop non-numerical and NaN values
data = data.dropna()

# Drop outliers using IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Data normalization
scaler = StandardScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data_normalized, test_size=0.2, random_state=42)

# Save the datasets
train_data.to_csv('dataset02_training.csv', index=False)
test_data.to_csv('dataset02_testing.csv', index=False)

# Define the target variable and features
X_train = train_data[['x']]  
y_train = train_data['y']     

# Add a constant to the model (intercept)
X_train = sm.add_constant(X_train)

# Fit the OLS model
model = sm.OLS(y_train, X_train).fit()

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot for training data
plt.scatter(train_data['x'], train_data['y'], color='orange', label='Training Data')
# Scatter plot for testing data
plt.scatter(test_data['x'], test_data['y'], color='blue', label='Testing Data')

# Red line plot for OLS predictions
plt.plot(train_data['x'], model.predict(X_train), color='red', label='OLS Fit')

plt.title('Scatter Plot of x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('UE_04_App2_ScatterVisualizationAndOlsModel.pdf')
plt.close()

# Create a box plot
plt.figure(figsize=(10, 6))
data_normalized.boxplot()
plt.title('Box Plot of All Dimensions')
plt.savefig('UE_04_App2_BoxPlot.pdf')
plt.close()

# Create diagnostic plots
diagnostic = LinearRegDiagnostic(model)
diagnostic(plot_context='seaborn-v0_8-paper')
plt.savefig('UE_04_App2_DiagnosticPlots.pdf')
plt.close()