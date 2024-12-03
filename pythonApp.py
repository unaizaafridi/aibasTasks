import pandas as pd
import statsmodels.api as sm


# Load the dataset
df = pd.read_csv('dataset01.csv')

y = df['y']
x = df['x']

# Determine and print the number of data entries of the column called 'y'
print("Number of data entries of the column 'y':", len(y))

# Determine and print the mean of the column called 'y'
print("Mean of the column 'y':", y.mean())

# Determine and print the standard deviation of the column called 'y'
print("Standard deviation of the column 'y':", y.std())

# Determine and print the variance of the column called 'y'
print("Variance of the column 'y':", y.var())

# Determine and print the min and max of the column called 'y'
print("Min of the column 'y':", y.min())
print("Max of the column 'y':", y.max())

# Determine and Fit the OLS model
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())

with open('OLS_model.txt', 'w') as file:
    file.write(str(result.summary()))

