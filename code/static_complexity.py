import numpy as np

def calculate_static_complexity(Y):
    Yref = 1
    emu = 0.5
    thetas = -(5 * ((1/27) * np.log2(1/27)) + ((2/27) * np.log2(2/27)) + ((20/27) * np.log2(20/27)))
    a = np.min(np.abs(Y - Yref))
    b = emu * np.max(np.abs(Y - Yref))
    c = np.abs(Y - Yref)
    r = (a + b) / (c + b)
    C0 = 0.1 * np.sum(r)
    Cs = 100 * C0 * thetas
    return Cs

# 示例数据
Y = np.array([0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0.5])
Cs = calculate_static_complexity(Y)
print("Static Complexity:", Cs)
