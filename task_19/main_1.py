import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data-export.csv")

df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

numeric_columns = [
    'Users', 'Sessions', 'Engaged sessions',
    'Average engagement time per session', 'Engaged sessions per user',
    'Events per session', 'Engagement rate', 'Event count'
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

half_index = len(df) // 2
df['Theme'] = ['Light'] * half_index + ['Dark'] * (len(df) - half_index)


light = df[df['Theme'] == 'Light']['Engagement rate'].dropna()
dark = df[df['Theme'] == 'Dark']['Engagement rate'].dropna()


t_stat, p_value = ttest_ind(light, dark, equal_var=False)


mean_light = light.mean()
mean_dark = dark.mean()


print("T-statistic:", t_stat)
print("P-value:", p_value)
print("Mean Engagement Rate (Light):", mean_light)
print("Mean Engagement Rate (Dark):", mean_dark)


alpha = 0.05
if p_value < alpha:
    print("Відкидаємо нульову гіпотезу: є статистично значуща різниця.")
else:
    print("Немає підстав відкидати нульову гіпотезу: різниця незначуща.")

sns.boxplot(data=df, x='Theme', y='Engagement rate')
plt.title("Engagement Rate by Theme")
plt.show()
