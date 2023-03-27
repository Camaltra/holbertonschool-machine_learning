#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bin_edges = np.arange(0, 101, 10)
plt.hist(student_grades, bins=bin_edges, edgecolor='black')
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
