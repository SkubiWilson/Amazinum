import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

df = pd.read_csv('insurance.csv')

print("First 5 rows:")
print(df.head())

print("Descriptive Statistics:")
print(df.describe(include='all'))

print("Missing Values:")
print(df.isnull().sum())

numeric_cols = ['age', 'bmi', 'children', 'charges']
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers


report = sv.analyze(df)
report.show_html("sweetviz_report.html")
print("Звіт sweetviz збережено як sweetviz_report.html")

charges_outliers = detect_outliers_iqr(df, 'charges')
print(f"Found {len(charges_outliers)} outliers in 'charges'")

print("\nDiscussion:")
print("'charges' має велику кількість викидів, ймовірно через високі медичні витрати пацієнтів із хронічними хворобами.")
print("'bmi' також містить деякі викиди (> 45), що може свідчити про ожиріння.")
print("Інші змінні мають відносно нормальні розподіли.")

df['is_outlier_charges'] = df['charges'].isin(charges_outliers['charges'])

print("Додано колонку 'is_outlier_charges'. Кількість позначених викидів:", df['is_outlier_charges'].sum())

