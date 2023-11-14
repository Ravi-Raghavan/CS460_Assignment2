import argparse
import numpy as np
import utils_p1 as utl
from utils_p1 import linear_search, kd_tree, k_nearest_neighbors
import matplotlib.pyplot as plt
import time

def main():
    # parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Naive Nearest Neighbor Search')
    parser.add_argument('--target', nargs=2, type=float, help='robot configuration')
    parser.add_argument('--k', type=int, help='nearest neighbors to be reported')
    parser.add_argument('--configs', help='random configurations list')
    args = parser.parse_args()
    target= args.target
    k = args.k
    configs = args.configs 

    # initialize robot
    lengths = [0.4,0.25]
    configs_list = np.load(configs, allow_pickle = True) 
    arm = utl.RobotArm('None', lengths, [target[0],target[1]], joint_radius=0.05, link_width=0.16611) #link_width is 1, 0.16611 makes it appear correctly
    
    # Find k nearest neighbors using linear search 
    timea = time.time()
    linear_neighbors = linear_search(target, k, configs_list)
    timeb = time.time()
    print("linear time:", timeb-timea)

    # Find k neighbors using kd tree
    tree = kd_tree(configs_list.tolist())
    time1 = time.time()
    kd_neighbors = k_nearest_neighbors(tree, target, k)
    time2 =time.time()
    print("time", time2-time1)

    #plot the cofigurations 
    arm.plot('black') 
    arm.plot_configs(kd_neighbors)
    '''
    FOR KD-TREE PLOT: arm.plot_configs(kd_neighbors)
    '''
    utl.plt.show()

if __name__ == "__main__":
    main()