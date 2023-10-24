import argparse
import numpy as np
import utils_p1 as utl
import time

# search for closest k neighbors using LINEAR approach
def linear_search(arm, link_lengths, target, k, configuration_list):
    # store distances between configurations and target
    distances = [] # stores (distance, joint angle 1, joint angle 2)

    # for each configuration, compute distance to target
    for configuration in configuration_list:
        # compute distance and add to arr ay
        neighbor = np.array(configuration[0],configuration[1])
        distance = compute_distance(arm, link_lengths, target, neighbor)
        distances.append(np.array(distance, configuration[0], configuration[1]))

    # sort array in ascending order based on distance
    sorted_distances = sorted(distances, key = lambda x:x[0]) 
    return sorted_distances[0:k]

# uses forward kinematics to calculate distance between 2 joint-angle configurations
# based on end-effect position
def compute_distance(lengths, target, neighbor): 
    # length of links
    a = lengths[0]
    b = lengths[1]

    # calculate position of target end-effector
    x_target = a*np.cos(target[0]) + b*np.cos(target[0]+target[1])
    y_target = a*np.sin(target[0]) + b*np.sin(target[0]+target[1])

    # calculate posiiton of neighbor end-effector
    x_neighbor = a*np.cos(neighbor[0]) + b*np.cos(neighbor[0]+neighbor[1])
    y_neighbor = a*np.sin(neighbor[0]) + b*np.cos(neighbor[0]+neighbor[1])

    # return distance between end-effectors
    return np.sqrt((x_target-x_neighbor)**2+(y_target-y_neighbor)**2)

# find closest neighbors using KD-TREE 
def kd_tree_search(arm, target, k, configs):
    distances = []
    sorted_distances = []
    # read all configurations in and create KD-Tree

    # return closest neighbors
    return sorted_distances[0:k]

# compares runtimes of KD tree and linear search given a list of configurations
def comparison(arm, link_lengths, target, k, configurations_list):
    # compute linear search time
    linear_start = time.time()
    linear_search(arm, link_lengths, target, k, configurations_list)
    linear_end = time.time()
    linear = linear_start - linear_end

    # compute KD-Tree search time
    kd_start = time.time()
    kd_tree_search(arm, link_lengths, target, k, configurations_list)
    kd_end = time.time()
    kd_tree = kd_end - kd_start

    # print search times 
    print(linear)
    print(kd_tree)


def main():
    # parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Naive Nearest Neighbor Search')
    parser.add_argument('--target', nargs=2, type=str, help='robot configuration')
    parser.add_argument('--k', type=int, help='nearest neighbors to be reported')
    parser.add_argument('--configs', help='random configurations list')
    args = parser.parse_args()
    target_config = args.target
    neighbors = args.k
    configs_list = args.configs

    # initialize robot
    lengths = [0.3,0.15]
    arm = utl.RobotArm(map, [0.3,0.15], [target_config[0], target_config[1]], joint_radius=0.05, link_width=0.1)

    # identidy closest k neighbors to target
    neighbors = linear_search(arm, lengths, target_config, neighbors, configs_list)
    
    # plot the cofigurations
    arm.plot_configs(neighbors)

if __name__ == "__main__":
    main()