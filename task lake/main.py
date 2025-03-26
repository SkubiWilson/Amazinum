import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
random_num = np.random.randint(1, 101, 20)

print(random_num)

lakes = []
for i in range(1, len(random_num) -1):
    if random_num[i] < random_num[i - 1] and random_num[i] < random_num [i + 1]:
        left = max(random_num[:i])
        right = max(random_num[i +1:])
        water_level = min(left, right)
        if water_level > random_num[i]:
            deph = water_level - random_num[i]
            lakes.append((i, deph))


plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), random_num, marker='o', linestyle='-', color='b', label="Висоти")

for i, deph in lakes:
    plt.scatter(i + 1, random_num[i], color='cyan', s=100, label="Озеро" if i == lakes[0][0] else "")

deepest = max(lakes, key= lambda x: x[1]) if lakes else None

print(f"Найглибше озеро: {deepest[0] + 1} з глибиною {deepest[1]}")

plt.scatter(deepest[0] + 1, random_num[deepest[0]], color="red", s=150, label="Найглибше озеро")

plt.xlabel("Порядковий номер точки")
plt.ylabel("Висота")
plt.title("Профіль місцевості (2D-гори)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.show()
