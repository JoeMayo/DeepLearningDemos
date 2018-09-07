import csv
import math
import random

def main():
    with open('xor_train.csv', 'w') as tstFile:
        tstWriter = csv.writer(tstFile)
        for i in range(1000):
            x1 = random.random()
            x2 = random.random()
            y = round(x1) ^ round(x2)
            tstWriter.writerow([x1, x2, y])

    with open('xor_test.csv', 'w') as tstFile:
        tstWriter = csv.writer(tstFile)
        for i in range(100):
            x1 = random.random()
            x2 = random.random()
            y = round(x1) ^ round(x2)
            tstWriter.writerow([x1, x2, y])

if __name__ == '__main__':
    main()