import numpy as np
import matplotlib.pyplot as plt


def calculate_dynamic_complexity(t):
    G = 1e-6
    k1 = 1
    k2 = 0.5
    mp = 1500
    mq = 2000

    thetap = np.arctan((11 - 9 * t) / (17 - 10 * t))
    thetaq = np.arctan((17 - 10 * t) / (11 - 9 * t))
    Mp = 1500 * (1 + (10 * np.cos(thetap)) ** 2)
    Mq = 2000 * (1 + (9 * np.cos(thetaq)) ** 2)
    R = np.exp(0.5 * 10 * np.cos(thetaq) / (10 * np.cos(thetap) + 9 * np.cos(thetaq)))
    S = np.sqrt((11 - 9 * t) ** 2 + (17 - 10 * t) ** 2)
    Cd = np.abs(G * R * Mp * Mq / S)

    return t, Cd


# 示例数据
t = np.arange(0, 3.1, 0.1)
t, Cd = calculate_dynamic_complexity(t)

plt.plot(t, Cd, 'b-', linewidth=2)
plt.title('Dynamic Complexity at Cross Intersection')
plt.xlabel('Time (s)')
plt.ylabel('Complexity')
plt.show()
