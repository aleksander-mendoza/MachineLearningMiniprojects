#!/bin/
import os


for txt in os.listdir('txt'):
    with open(txt) as file:
        words = {}
        for line in file:
            for word in line.split():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
