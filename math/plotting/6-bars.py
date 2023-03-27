#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
data = [fruit[i, :] for i in range(len(fruit))]
names = ["Farrah", "Fred", "Felicia"]
print(data)

plt.bar(names, data[0], color="r", label="apples", width=0.5)
plt.bar(names, data[1], color="yellow",
        bottom=data[0], label="bananas", width=0.5)
plt.bar(names, data[2], color="#ff8000",
        bottom=data[0]+data[1], label="oranges", width=0.5)
plt.bar(names, data[3], color="#ffe5b4",
        bottom=data[0]+data[1]+data[2], label="peaches", width=0.5)
plt.ylim([0, 80])
plt.legend(loc="upper right")
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.show()
