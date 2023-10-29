### Problem 2: Motion Planning for a Rigid Body
## 2.2: Nearest neighbors with linear search approach
from utils_p2 import RigidBody

import argparse
import matplotlib.pyplot as plt
import numpy as np

#Set up Plot Information
f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

#Load in map of rigid polygons
parser = argparse.ArgumentParser(description="Receive Command Line Arguments for Nearest Neighbor Problem")
parser.add_argument("--target", nargs = 3, type = int, help = "Target Configuration")
parser.add_argument("--k", nargs = 1, type = int, help = "Target Configuration")
parser.add_argument("--configs", nargs = 1, type = str, help = "List of Random Configurations")
args = parser.parse_args()

#Get the values from the parser
if args.target == None or len(args.target) != 3: exit(0)
target_configuration = np.array([args.target[0], args.target[1], args.target[2]])

if args.k == None or len(args.k) != 1: exit(0)
k = args.k[0]

if args.configs == None or len(args.configs) != 1: exit(0)
rigid_configs_file = args.configs[0]

diameter = np.linalg.norm(np.array([0.1, 0.2])) #Diameter of Circle in which the rigid body is inscribed
radius = 0.5 * diameter #Radius of Circle in which the rigid body is inscribed

#define our Distance Function
def D(point):
    #Flatten
    point = point.flatten()
    
    #Calculate Euclidean Distance Transitionally
    dt = np.linalg.norm(point[:-1])
    
    #Calculate Rotational Distance
    angle = point[-1]
    angle = angle if angle > 0 and angle < 180 else 180 - angle
    angle = np.deg2rad(angle)
    dr = radius * np.abs(angle)
    
    return 0.7 * dt + 0.3 * dr

configurations = np.load(rigid_configs_file, allow_pickle= True)
A = configurations - target_configuration
neighbor_distances = np.apply_along_axis(func1d = D, axis = 1, arr = A)

print(f"Shape Information for Sanity: {A.shape}, {neighbor_distances.shape}")
print("Value of K is ", k)

k_neighbor_indices = np.argsort(neighbor_distances)[:k]

#Define Rigid Body and plot Configurations
rigid_body = RigidBody(f, ax, None)
rigid_body.plot_configuration(target_configuration, color = "black")
rigid_body.plot_configuration(configurations[k_neighbor_indices[0]], color = "red")
rigid_body.plot_configuration(configurations[k_neighbor_indices[1]], color = "green")
rigid_body.plot_configuration(configurations[k_neighbor_indices[2]], color = "blue")

#Plot everything else
for neighbor_index in k_neighbor_indices[3:k]:
    rigid_body.plot_configuration(configurations[neighbor_index], color = "yellow")

plt.show()