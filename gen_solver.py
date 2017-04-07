from solver_params import SolverParameters
from math import ceil



def get_dataset_size(path=None, dataset='', phase='', mode=''):
    # path parameter has greater priority
    if path == None:
        with open("../local_data/{}/{}/{}_size.txt".format(dataset, mode, phase)) as f:
             return int(f.read()) # we assume that the file contains only 1 integer value written as a string
    else:
         with open(path) as f:
             return int(f.read()) # we assume that the file contains only 1 integer value written as a string

       

def GenSingleNetSolver(dataset, mode, args):

    train_size = get_dataset_size(dataset=dataset, phase="train", mode=mode)
    test_size = get_dataset_size(dataset=dataset, phase="test", mode=mode) 

    test_iter = ceil(test_size / float(args.batch_size))
    epoch_sz = ceil(train_size / float(args.batch_size))


    directory = "{}/experiment_{}/{}/{}".format(args.proto_pref,  args.EXPERIMENT_NUMBER, dataset, mode)
    train_path = "{}/train.prototxt".format(directory)
    test_path = "{}/test.prototxt".format(directory)


    print("Generating solver")
    print("{} {}\n".format(dataset, mode))    
    #why do I need this path? 
    safe_mkdir('{}/{}/{}/'.format(args.proto_pref,dataset,mode))
    snap_pref = "{}/experiment_{}/{}/{}/snap".format(args.snap_pref,args.EXPERIMENT_NUMBER, dataset, mode) 

    


    p = SolverParameters(train_net_path=train_path, test_net_path=test_path, test_iter=test_iter, lr=args.learning_rate,
                         train_epoch_sz=epoch_sz, n_epoch=args.epoch, test_epoch=args.test_frequency,
                         snap_pref=snap_pref, snap_epoch=args.snap_epoch, step_epoch=args.step_epoch, gamma=args.gamma)  



    
    with open('{}/solver.prototxt'.format(directory), 'w') as f:
        f.write(p.solvet_txt) 
        print(p.solvet_txt)




def CommiteeSolver(dataset, args):

    train_size = get_dataset_size(dataset=dataset, phase="train", mode="orig")
    test_size = get_dataset_size(dataset=dataset, phase="test", mode="orig") 

    test_iter = ceil(test_size / float(args.batch_size))
    epoch_sz = ceil(train_size / float(args.batch_size))


    directory = "{}/experiment_{}/{}/commitee".format(args.proto_pref,  args.EXPERIMENT_NUMBER, dataset)
    train_path = "{}/train.prototxt".format(directory)
    test_path = "{}/test.prototxt".format(directory)


    print("Generating solver")
    print("{}\n".format(dataset))     
    # safe_mkdir('{}/{}/'.format(args.proto_pref,dataset))
    snap_pref = "{}/experiment_{}/{}/snap".format(args.snap_pref,args.EXPERIMENT_NUMBER, dataset) 

    


    p = SolverParameters(train_net_path=train_path, test_net_path=test_path, test_iter=test_iter, lr=args.learning_rate,
                         train_epoch_sz=epoch_sz, n_epoch=args.epoch, test_epoch=args.test_frequency,
                         snap_pref=snap_pref, snap_epoch=args.snap_epoch, step_epoch=args.step_epoch, gamma=args.gamma)  



    
    with open('{}/solver.prototxt'.format(directory), 'w') as f:
        f.write(p.solvet_txt) 
        print(p.solvet_txt)