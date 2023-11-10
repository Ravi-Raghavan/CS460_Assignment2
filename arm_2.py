import argparse
import numpy as np
import utils_p1 as utl
from utils_p1 import compute_distance, compute_norm
import matplotlib.pyplot as plt
import time
from queue import PriorityQueue

######################### LINEAR SEARCH IMPLEMENTATION ######################
def linear_search(target, k, configurations):
    # store distances between configurations and target
    distances = [] # stores (distance, joint angle 1, joint angle 2)
    i = 0

    start = time.time()

    # for each configuration, compute distance to target
    for configuration in configurations:
        i =i +1
        # compute distance and add to arr ay
        angular_difference = compute_distance(target, configuration)
        distance_to_target = compute_norm(angular_difference)
        distances.append((distance_to_target, configuration))

    # sort array in ascending order based on distance
    sorted_distances = sorted(distances, key = lambda x:x[0])

    #end = time.time()
    #timer = end-start
    #print("linear time: ")
   # print(timer)
    # create array of just configurations 
    closest_neighbors = []
    for i in range(k):
        _, config = sorted_distances[i]
        closest_neighbors.append(config)
    return closest_neighbors

######################### KD-TREE IMPLEMENTATION ######################
# KD-Tree construction
class Node:
    def __init__(self, point, split_dim):
        self.point = point
        self.split_dim = split_dim
        self.left = None
        self.right = None

def kd_tree(points, depth=0):
    if len(points) == 0: #if none, all nodes added to tree
        return None

    k = len(points[0]) 
    split_dim = depth % k

    points.sort(key= lambda x : x[split_dim])

    median = len(points) // 2

    node = Node(points[median], split_dim)
    node.left = kd_tree(points[:median], depth + 1)
    node.right = kd_tree(points[median + 1:], depth + 1)

    return node

# Search KD-Tree
def k_nearest_neighbors(root, query_point, k):
    pq = PriorityQueue()
    
    def search(node):
        nonlocal pq  
        if node is None:
            return
        
        ang_dist = compute_distance(query_point, node.point)
        dist = compute_norm(ang_dist)
        if len(pq.queue) < k:
            pq.put((-dist, node.point))
        else:
            if -dist > pq.queue[0][0]:
                pq.get()
                pq.put((-dist, node.point))

        # Recursive search on child nodes
        split_dim = node.split_dim
        diff = utl.normalize_angle(query_point[split_dim]) - utl.normalize_angle(node.point[split_dim])
        if diff <= np.pi:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        search(first)

        if len(pq.queue) < k or -abs(diff) < -pq.queue[0][0]: 
            search(second)
    
    start = time.time()
    search(root)
    end = time.time()

    timer = end-start
    print("kd-tree time: ")
    print(timer)

    neighbors = [point for dist, point in sorted(pq.queue, key=lambda x: x[0], reverse = True)]
    return neighbors

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
    lengths = [0.3,0.15]
    configs_list = np.load(configs, allow_pickle = True) 
    arm = utl.RobotArm('None', lengths, [target[0],target[1]], joint_radius=0.05, link_width=0.1)
    
    # Find k nearest neighbors using linear search 
    linear_neighbors = linear_search(target, k, configs_list)

    # Find k neighbors using kd tree
    tree = kd_tree(configs_list.tolist())
    kd_neighbors = k_nearest_neighbors(tree, target, k)

    #plot the cofigurations 
    arm.plot('black') 
    arm.plot_configs(kd_neighbors)
    '''
    FOR KD-TREE PLOT: arm.plot_configs(kd_neighbors)
    '''
    utl.plt.show()

if __name__ == "__main__":
    main()