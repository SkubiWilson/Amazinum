import math
import random

def simulate():
    stakan = random.randint(1, 100)
    stakan_count = int(input("Кількість стаканів: "))
    level = int(input("Кількість поверхів: "))

    floors = [random.randint(1, 2000) for _ in range(level)]
    floors_sort = sorted(floors)

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

    print(f"Мінімальна кількість кидків для {level} поверхів і {stakan_count} стаканів це: {min_kroky}")
    print(f"Стратегія кидків: {strategy}")

for _ in range(3):
    simulate()
