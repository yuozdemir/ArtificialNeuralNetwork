import numpy as np
import statistics


def act(x_):
    return 1 / (1 + np.exp(-x_))


x = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 1, 1, 0],
              [0, 0, 1, 1, 1],
              [0, 1, 0, 0, 0],
              [0, 1, 0, 0, 1],
              [0, 1, 0, 1, 0],
              [0, 1, 0, 1, 1],
              [0, 1, 1, 0, 0],
              [0, 1, 1, 0, 1],
              [0, 1, 1, 1, 0],
              [0, 1, 1, 1, 1],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 1, 0],
              [1, 0, 0, 1, 1],
              [1, 0, 1, 0, 0],
              [1, 0, 1, 0, 1],
              [1, 0, 1, 1, 0],
              [1, 0, 1, 1, 1],
              [1, 1, 0, 0, 0],
              [1, 1, 0, 0, 1],
              [1, 1, 0, 1, 0],
              [1, 1, 0, 1, 1],
              [1, 1, 1, 0, 0],
              [1, 1, 1, 0, 1],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1]])

y = np.array([[0,
               1,
               1,
               0,
               1,
               0,
               0,
               1,
               1,
               0,
               0,
               1,
               0,
               1,
               1,
               0,
               1,
               0,
               0,
               1,
               0,
               1,
               1,
               0,
               0,
               1,
               1,
               0,
               1,
               0,
               0,
               1]]).T

input_size = x.shape[1]
hidden_size = 4
output_size = 1
alpha = 0.1

w1 = np.random.randn(input_size, hidden_size)
w2 = np.random.randn(hidden_size, output_size)

iteration = 0

while True:

    z1 = np.dot(x, w1)
    a1 = act(z1)
    z2 = np.dot(a1, w2)
    Y = act(z2)

    delta2 = (Y - y) * (Y * (1 - Y))
    delta1 = np.dot(delta2, w2.T) * (a1 * (1 - a1))
    w2 -= alpha * np.dot(a1.T, delta2)
    w1 -= alpha * np.dot(x.T, delta1)

    K = []
    for i in Y:
        K.append(i[0])

    L = []
    for i in K:
        if i > statistics.median(K):
            L.append(1)
        else:
            L.append(0)

    T = 0
    for i in range(len(L)):
        if L[i] == y[i]:
            T += 1

    if T == 32:
        print(T)
        break

    print(iteration)
    iteration += 1

z1 = np.dot(x, w1)
a1 = act(z1)
z2 = np.dot(a1, w2)
Y = act(z2)

print("Output after training...")
print(Y)
