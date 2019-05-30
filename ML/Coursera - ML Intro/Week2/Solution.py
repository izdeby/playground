
import torch

#Read data
x = []
y = []
with open("ex1/ex1data1.txt", "r") as ins:
    for line in ins:
        print(line)
        x.append(line.split(',')[0])
        y.append(line.split(',')[1])

