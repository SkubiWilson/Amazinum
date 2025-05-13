import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("test.csv")

print("First 5 rows:")
print(df.head())

survey_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness'
]

survey_df = df[survey_cols]

print("Missing values:")
print(survey_df.isnull().sum())

survey_df.dropna(inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(survey_df)

kmo_all, kmo_model = calculate_kmo(scaled_data)
bartlett_test, p_value = calculate_bartlett_sphericity(scaled_data)
print(f"\nKMO Measure: {kmo_model:.2f}")
print(f"Bartlett's test p-value: {p_value:.5f}")

fa = FactorAnalyzer(rotation=None)
fa.fit(scaled_data)
eigenvalues, _ = fa.get_eigenvalues()

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
plt.axhline(y=1, color='r', linestyle='--')
plt.title("Scree Plot")
plt.xlabel("Factor Number")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()

n_factors = 4

fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(scaled_data)
loadings = pd.DataFrame(fa.loadings_, index=survey_cols, columns=[f'Factor {i+1}' for i in range(n_factors)])

print("\nFactor Loadings (Varimax):")
print(loadings.round(2))

for i in range(n_factors):
    print(f"\nFactor {i+1} top items:")
    print(loadings.iloc[:, i].abs().sort_values(ascending=False).head(5))
