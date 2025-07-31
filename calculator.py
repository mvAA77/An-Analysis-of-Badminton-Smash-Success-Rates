import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare

list = ["Smash"]
with open(f"stroke_types.csv", mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row["StrokeType"] not in list:
            list.append(row["StrokeType"])
###print(list)

count = [0] * len(list)

for i in range(1, 10):
    with open(f"last_stroke_types_{i}.csv", mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            stroke_type = row["LastStrokeType"]
            ##print(f"Found stroke: {stroke_type}")  # Debug line
            if stroke_type in list:
                count[list.index(stroke_type)] += 1
###print(count)


categories = []
for i in range(len(list)):
    categories.append(list[i][:3])
values = count

plt.bar(categories, values)            
plt.xlabel('Stroke Types')

plt.show()


result = chisquare(count)

# Print the results in a readable format
print("Chi-Square Test Results:")
print(f"Test Statistic (χ²): {result.statistic:.2f}")
print(f"P-value: {result.pvalue:.4f}")
