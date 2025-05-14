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


print("Перевірка статистичної гіпотези щодо впливу теми інтерфейсу на Engagement Rate\n")

print("Формулювання гіпотез:")
print("Нульова гіпотеза (H₀): Середній Engagement Rate однаковий для користувачів з темною і світлою темою.")
print("Альтернативна гіпотеза (H₁): Середній Engagement Rate відрізняється між користувачами з темною і світлою темою.\n")

print("Результати t-тесту:")
print(f"T-статистика: {t_stat:.4f}")
print(f"P-значення: {p_value:.4f}")
print(f"Середній Engagement Rate (Light): {mean_light:.4f}")
print(f"Середній Engagement Rate (Dark): {mean_dark:.4f}\n")

alpha = 0.05
print(f"Рівень значущості (α): {alpha}")
if p_value < alpha:
    print("Висновок: P-значення < α → Відкидаємо нульову гіпотезу.")
    print("Є статистично значуща різниця між Engagement Rate у темній та світлій темах.")
else:
    print("ℹВисновок: P-значення ≥ α → Немає підстав відкидати нульову гіпотезу.")
    print("Статистично значущої різниці між Engagement Rate не виявлено.")



sns.boxplot(data=df, x='Theme', y='Engagement rate')
plt.title("Engagement Rate by Theme")
plt.show()

