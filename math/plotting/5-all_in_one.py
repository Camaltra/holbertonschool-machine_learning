#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.rc('axes', titlesize="x-small")
plt.rc('axes', labelsize="x-small")

fig, ax = plt.subplot_mosaic([["upper left", "upper right"],
                              ["middle left", "middle right"],
                              ["lower", "lower"]],
                             layout="constrained")
fig.tight_layout(pad=2.5)
fig.suptitle("All in One")

ax["upper left"].plot(y0, 'r')
ax["upper left"].set_xlim([0, 10])


ax["upper right"].scatter(x1, y1, c='magenta', s=10)
ax["upper right"].set_xlabel("Height (in)")
ax["upper right"].set_ylabel("Height (lbs)")
ax["upper right"].set_title("Men's Height vs Weight")

ax["middle left"].plot(x2, y2)
ax["middle left"].set_yscale('log')
ax["middle left"].set_xlim([x2[0], x2[-1]])
ax["middle left"].set_xlabel("Time (years)")
ax["middle left"].set_ylabel("Fraction Remaining")
ax["middle left"].set_title("Exponential Decay of C-14")

ax["middle right"].plot(x3, y31, "r--", label="C-14")
ax["middle right"].plot(x3, y32, "g", label="Ra-226")
ax["middle right"].legend()
ax["middle right"].set_ylim(0, 1)
ax["middle right"].set_xlim(x3[0], x3[-1])
ax["middle right"].set_xlabel("Time (years)")
ax["middle right"].set_ylabel("Fraction Remaining")
ax["middle right"].set_title("Exponential Decay of Radioactive Elements")

bin_edges = np.arange(0, 101, 10)
ax["lower"].hist(student_grades, bins=bin_edges, edgecolor='black')
ax["lower"].set_xlim([0, 100])
ax["lower"].set_ylim([0, 30])
ax["lower"].set_xlabel('Grades')
ax["lower"].set_ylabel('Number of Students')
ax["lower"].set_title('Project A')

fig.show()
