import math
import random

stakan = random.randint(1, 100)
stakan_count = int(input("Кількість стаканів: "))
level = int(input("Кількість поверхів: "))
print(f"Міцність стакану: {stakan}")

floors = [random.randint(1, 200) for _ in range(level)]
floors_sort = sorted(floors)

print("Навантаження з кожного поверху (від найменшого до найбільшого):")
for i, val in enumerate(floors_sort, 1):
    print(f"{i}: {val} балів")

def broke(value):
    return value > stakan

def find_break_floor():
    for i, val in enumerate(floors_sort):
        if broke(val):
            return i
    return -1

break_index = find_break_floor()

def krok(n, k):
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    m = 0
    while dp[m][k] < n:
        m += 1
        for j in range(1, k + 1):
            dp[m][j] = dp[m - 1][j - 1] + dp[m - 1][j] + 1
    return m

def throw_strategy(n, k):
    step = krok(n, k)
    throws = []
    floor = 0
    for i in range(step, 0, -1):
        floor += i
        if floor > n:
            break
        throws.append(floor)
    return throws

min_kroky = krok(level, stakan_count)
strategy = throw_strategy(level, stakan_count)

if break_index == -1:
    print("Стакан не розбився навіть з останнього поверху.")
else:
    print(f"Стакан розбився з поверху: {break_index + 1}")
    print(f"Максимальний безпечний поверх: {break_index}")
print(f"Оптимальна стратегія кидків для {level} поверхів і {stakan_count} стаканів: {min_kroky}")

print("Рекомендовані поверхи для кидків (з номерами кроків):")
for i, floor in enumerate(strategy, 1):
    print(f"Крок {i}: кидати з поверху {floor}")