#!/usr/bin/env python3

import random
import sys

number_of_samples = int(input())
lenght_of_sample = 100
number_of_tests = int(input())
labels = []
print(0.15)
print(number_of_samples)
for i in range(number_of_samples):
    s = 0
    for j in range(lenght_of_sample):
        number = random.randint(0, 10000)
        s += number
        print(number)
    labels.append(s % 16)
for i in range(number_of_samples):
    print(labels[i])
for i in range(0,number_of_tests):
    s = 0
    for j in range(lenght_of_sample):
        number = random.randint(0, 10000)
        s += number
        print(number)
    print(s % 16, file=sys.stderr)
