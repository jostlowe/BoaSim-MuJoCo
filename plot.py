import matplotlib.pyplot as plt

import csv
import numpy as np

def read_csv(filename):
    with open(filename, newline='') as f_input:
        return [list(map(float, row)) for row in csv.reader(f_input)]

data = np.array(read_csv("data.csv"))

plt.plot(data[:, 1], data[:, 2])
plt.gca().set_aspect('equal')
plt.show()

#plt.plot(data[:, 0], abs(data[:, 4:]).max(axis=1))
#plt.plot(data[:, 0], abs(data[:, 4:]).mean(axis=1))
#plt.show()