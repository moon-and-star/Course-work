#!/usr/bin/env python
#этот скрипт нужен для создания усеченных gt файлов, в которых будут содержаться записи только о знаках тех класссов,
# в котоых количество элементов не ниже определенного порога (суммарно по train и test)

if __name__ == '__main__':
    threshold = 1000

    rootpath = '../global_data/Traffic_signs/RTSD/classification'
    class_size = [0] * 116
    print("lol")
    for phase in ["train", "test"]:
        print("ololo")
        with open("{}/gt_{}_full.txt".format(rootpath, phase)) as f:
            for line in f.readlines():
                s = line.split("/")[0]
                class_size[int(s)] += 1

    print(class_size)
    for phase in ["train", "test"]:
        with open("{}/gt_{}_full.txt".format(rootpath, phase)) as f:
            with open("{}/gt_{}.txt".format(rootpath, phase), 'w') as out:
                for line in f.readlines():
                    s = line.split("/")[0]
                    if class_size[int(s)] >= threshold:
                        out.write(line)