import argparse
import numpy as np
import utils_p1 as utl
import matplotlib.pyplot as plt
import time
from queue import PriorityQueue

# search for closest k neighbors using LINEAR brute force approach
def linear_search(link_lengths, target, k, configurations):
    # store distances between configurations and target
    distances = [] # stores (distance, joint angle 1, joint angle 2)

    # for each configuration, compute distance to target
    for configuration in configurations:
        # compute distance and add to arr ay
        distance_to_target = compute_distance(link_lengths, target, configuration)
        distances.append((distance_to_target, configuration))

    # sort array in ascending order based on distance
    sorted_distances = sorted(distances, key = lambda x:x[0]) 

    # create array of just configurations 
    closest_neighbors = []
    for i in range(k):
        _, config = sorted_distances[i]
        closest_neighbors.append(config)

    return closest_neighbors

######################### KD- TREE IMPLEMENTATION ######################
# KD-Tree Node class
class Node:
    def __init__(self, point, split_dim):
        self.point = point
        self.split_dim = split_dim
        self.left = None
        self.right = None

# KD-Tree construction
def kd_tree(points, depth=0):
    if len(points) == 0:
        return None

    k = len(points[0])
    split_dim = depth % k

    points.sort(key=lambda x: x[split_dim])

    median = len(points) // 2

    node = Node(points[median], split_dim)
    node.left = kd_tree(points[:median], depth + 1)
    node.right = kd_tree(points[median + 1:], depth + 1)
    return node

def k_nearest_neighbors(root, query_point, k):
    pq = PriorityQueue()

    def search(node):
        nonlocal pq  
        if node is None:
            return

        # Calculate distance
        lengths = [0.3,0.15]
        dist = compute_distance(lengths, query_point, node.point)

        if len(pq.queue) < k:
            pq.put((-dist, node.point))
        else:
            if -dist > pq.queue[0][0]:
                pq.get()
                pq.put((-dist, node.point))

        # Recursive search on child nodes
        split_dim = node.split_dim
        diff = query_point[split_dim] - node.point[split_dim]

        if diff <= 0:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        search(first)

        if len(pq.queue) < k or -diff < -pq.queue[0][0]:
            search(second)

    search(root)
    neighbors = [point for dist, point in sorted(pq.queue, key=lambda x: x[0], reverse = True)]

    return neighbors

##########################################################

# uses forward kinematics to calculate distance between 2 joint-angle configurations
# based on end-effector position
def compute_distance(lengths, target, neighbor): 
    # length of links
    a = lengths[0]
    b = lengths[1]

    # calculate position of target end-effector
    x_target = a*np.cos(target[0]) + b*np.cos(target[0]+target[1])
    y_target = a*np.sin(target[0]) + b*np.sin(target[0]+target[1])

    # calculate posiiton of neighbor end-effector
    x_neighbor = a*np.cos(neighbor[0]) + b*np.cos(neighbor[0]+neighbor[1])
    y_neighbor = a*np.sin(neighbor[0]) + b*np.sin(neighbor[0]+neighbor[1])

    # return distance between end-effectors
    return np.sqrt((x_target-x_neighbor)**2+(y_target-y_neighbor)**2)

# compares runtimes of KD tree and linear search given a list of configurations
def comparison(arm, link_lengths, target, k, configurations_list):
    # compute linear search time
    linear_start = time.time()
    linear_search(arm, link_lengths, target, k, configurations_list)
    linear_end = time.time()
    linear = linear_start - linear_end

    # compute KD-Tree search time
    kd_start = time.time()
    k_nearest_neighbors(arm, link_lengths, target, k, configurations_list)
    kd_end = time.time()
    kd_tree = kd_end - kd_start

    # print search times 
    print(linear)
    print(kd_tree)

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

    # Find k nearest neighbors
    linear_neighbors = linear_search(lengths, target, k, configs_list)
    tree = kd_tree(configs_list.tolist())
    nearest_points = k_nearest_neighbors(tree, target, k)

    #plot the cofigurations 
    arm.plot('black')
    arm.plot_configs(nearest_points)
    #arm.plot_configs(linear_neighbors)
    utl.plt.show()

if __name__ == "__main__":
    main()