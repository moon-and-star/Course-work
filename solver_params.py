from math import ceil
import argparse
from util import safe_mkdir, gen_parser

solver_template = 
""" train_net: "{train_path}"
    test_net: "{test_path}"

    test_iter: {test_iter}
    test_interval: {test_interval}

    type : "{slvr_type}"
    base_lr: {lr}
    stepsize: {stepsize}

    lr_policy: "step"
    gamma: 0.1
    iter_size: 1

    momentum: 0.9
    weight_decay: 0.0005
    display: 1
    max_iter: {max_iter}
    snapshot: {snap_iter}
    snapshot_prefix: "{snap_pref}"
    solver_mode: GPU
    """


class SolverParameters(object):    
    def __init__(self, train_net_path, test_net_path,#architecture paths
                 test_iter, #number of iterations for test set 
                 max_iter=None, n_epoch=100, train_epoch_sz=None, 
                 test_epoch=1, test_interval=None, # how often to test (every n epoch/ every n iterations)
                 snap_pref="", snap_epoch=None, snap_iter=None,
                 slvr_type="Adam", lr = 1e-3, step_iter=None, step_epoch=1): 
        super(SolverParameters, self).__init__()

        test_iter = int(test_iter)
        #add fiction parameters to prevent use of train_epoch_sz if max iter specified
        snap_epoch_size = None
        step_epoch_size = None
        test_epoch_size = None
        #TRAINING LIMIT
        self.max_iter = max_iter
        if max_iter == None:
            self.max_iter = int(n_epoch * train_epoch_sz)
            snap_epoch_size = train_epoch_sz
            step_epoch_size = train_epoch_sz
            test_epoch_size = train_epoch_sz
        elif train_epoch_sz != None:
            self.max_iter = int(max_iter)
            print("WARNING: max_iter parameter has greater priority. train_epoch_sz will be ignored")


        #TEST FREQUENCY
        self.test_interval = test_interval
        if test_interval == None:
            if test_epoch_size == None:
                print('ERROR: expected test_interval parameter')
                exit()
            self.test_interval = int(test_epoch * test_epoch_size) 
        elif test_epoch != None:
            self.test_interval = int(test_interval)
            print("WARNING: test_interval parameter has greater priority. test_epoch will be ignored")
        

        #SNAPSHOT FREQUENCY
        self.snap_iter = snap_iter
        if snap_iter == None:
            if snap_epoch_size == None:
                print('ERROR: expected snap_iter parameter')
                exit()
            self.snap_iter = int(snap_epoch * snap_epoch_size)
        elif snap_epoch != None:
            self.snap_iter = int(snap_iter)
            print("WARNING: snap_iter parameter has greater priority. snap_epoch will be ignored")


        self.lr = lr
        self.step_iter = step_iter
        if step_iter == None:
            if step_epoch_size == None:
                print('ERROR: expected step_iter parameter')
                exit()
            self.step_iter = int(step_epoch * step_epoch_size)
        elif step_epoch != None:
            print("WARNING: step_iter parameter has greater priority. step_epoch will be ignored")



        self.solvet_txt = solver_template.format(
                          train_path=train_net_path, test_path=test_net_path,
                          snap_pref=snap_pref, test_iter=test_iter,
                          test_interval=self.test_interval, max_iter=self.max_iter,
                          snap_iter=self.snap_iter, stepsize=self.step_iter, 
                          slvr_type=slvr_type, 
                          lr=lr
                          )




def get_dataset_size(path=None, dataset='', phase='', mode=''):
    # path parameter has greater priority
    if path == None:
        with open("../local_data/{}/{}/{}_size.txt".format(dataset, mode, phase)) as f:
             return int(f.read()) # we assume that the file contains only 1 integer value written as a string
    else:
         with open(path) as f:
             return int(f.read()) # we assume that the file contains only 1 integer value written as a string

       


def gen_solver(dataset, mode, args):

    train_size = get_dataset_size(dataset=dataset, phase="train", mode=mode)
    test_size = get_dataset_size(dataset=dataset, phase="test", mode=mode) 

    test_iter = ceil(test_size / float(args.batch_size))
    epoch_sz = ceil(train_size / float(args.batch_size))


    directory = "{}/{}/{}".format(args.proto_pref, dataset, mode)
    train_path = "{}/train.prototxt".format(directory)
    test_path = "{}/test.prototxt".format(directory)


    print("Generating solver")
    print("{} {}\n".format(dataset, mode))     
    safe_mkdir('{}/{}/{}/'.format(args.proto_pref,dataset,mode))
    snap_pref = "{}/experiment_{}/{}/{}".format(args.snap_pref,args.EXPERIMENT_NUMBER, dataset, mode) 

    


    p = SolverParameters(train_net_path=train_path, test_net_path=test_path, test_iter=test_iter, lr=args.learning_rate,
                         train_epoch_sz=epoch_sz, n_epoch=args.epoch, test_epoch=args.test_frequency,
                         snap_pref=snap_pref, snap_epoch=args.snap_epoch, step_epoch=args.step_epoch)  



    
    with open('{}/{}/{}/solver.prototxt'.format(args.proto_pref,dataset, mode), 'w') as f:
        f.write(p.solvet_txt) 
        # print(p.solvet_txt)
    

