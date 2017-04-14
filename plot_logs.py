#!/usr/bin/env python
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

#experiment folder should contain 2 files: prefix.train and prefix.test
usage = "\nusage:    {} [EXPERIMENT_LOG_FOLDER_PATH] [COMMON_PREFIX_OF_LOGS]".format(sys.argv[0])
err = "\nTOO FEW ARGUMENTS!!!"

def launch():
    if len(sys.argv) < 3:
        print(err,usage)
        sys.exit()

    print("\n\nplotting logs ")
    path = sys.argv[1]
    prefix = sys.argv[2]
    print("path = {}      \nprefix = {}".format(path, prefix))
    for size in[(60, 40), (15,10)]:
        train_log = pd.read_csv("{}/{}.train".format(path, prefix))
        test_log = pd.read_csv("{}/{}.test".format(path, prefix))
        _, ax1 = subplots(figsize=size)
        ax2 = ax1.twinx()
        ax1.plot(train_log["NumIters"] , train_log["loss"], alpha=0.4)
        ax1.plot(test_log["NumIters"] , test_log["loss"], 'g')
        ax2.plot(test_log["NumIters"] , test_log["accuracy_1"], 'r')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('train loss')
        ax2.set_ylabel('test accuracy')

        # plt.show()

        plt.savefig('{}/plot_{}.png'.format(path, size))
    
    print("\n\n")




# matplotlib.use('Agg')
launch()

