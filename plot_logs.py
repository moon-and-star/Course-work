#!/usr/bin/env python
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

# train_log = pd.read_csv("./logs/CoNorm.txt.train")
# test_log = pd.read_csv("./logs/CoNorm.txt.test")

usage = "usage:    {} [EXPERIMENT_NUMBER] [NAME1, NAME2....]\n".format(sys.argv[0])
err = "\nTOO FEW ARGUMENTS!!!\n"
if len(sys.argv) == 1:
    print(err,usage)
    sys.exit()

num = sys.argv[1]

names = []
for i in range(2, len(sys.argv)):
    names.append(sys.argv[i])


# for name in ["CoNorm", 'imajust', 'histeq', 'AHE', 'orig']:
for name in names:
    # print("EXPERIMENT_NUMBER = ", num)
    print("      plotting logs for ", name)
    for size in[(60, 40), (15,10)]:
        train_log = pd.read_csv("./logs/{0}/{0}_{1}/{0}.txt.train".format(name, num))
        test_log = pd.read_csv("./logs/{0}/{0}_{1}/{0}.txt.test".format(name, num))
        _, ax1 = subplots(figsize=size)
        ax2 = ax1.twinx()
        ax1.plot(train_log["NumIters"] , train_log["loss"], alpha=0.4)
        ax1.plot(test_log["NumIters"] , test_log["loss"], 'g')
        ax2.plot(test_log["NumIters"] , test_log["accuracy_1"], 'r')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('train loss')
        ax2.set_ylabel('test accuracy')

        # plt.show()

        plt.savefig('logs/{0}/{0}_{1}/{0}_{2}.png'.format(name, num, size))