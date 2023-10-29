### Problem 2: Motion Planning for a Rigid Body
## 2.5: Implement a Probabilistic Road Map
from utils_p2 import RigidBody, RRT

import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

#Parse Arguments
parser = argparse.ArgumentParser(description="Receive Command Line Arguments for PRM Problem")
parser.add_argument("--start", nargs = 3, type = float, help = "Start Configuration")
parser.add_argument("--goal", nargs = 3, type = float, help = "Goal Configuration")
parser.add_argument("--map", nargs = 1, type = str, help = "Map of Rigid Polygons")
args = parser.parse_args()

rigid_polygons_file = args.map[0]
rigid_body = RigidBody(f, ax, rigid_polygons_file)

start = np.array([args.start[0], args.start[1], args.start[2]])
goal = np.array([args.goal[0], args.goal[1], args.goal[2]])

N = 1000
prm = PRM(N, rigid_body)

path_cost = prm.answer_query(start, goal)
print("Path Cost: ", path_cost)