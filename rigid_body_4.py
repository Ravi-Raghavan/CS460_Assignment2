### Problem 2: Motion Planning for a Rigid Body
## 2.4: Implement Rapidly exploring random tree
from utils_p2 import RigidBody, RRT

import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

#Parse Arguments
parser = argparse.ArgumentParser(description="Receive Command Line Arguments for RRT Problem")
parser.add_argument("--start", nargs = 3, type = float, help = "Start Configuration")
parser.add_argument("--goal", nargs = 3, type = float, help = "Goal Configuration")
parser.add_argument("--map", nargs = 1, type = str, help = "Map of Rigid Polygons")
args = parser.parse_args()

rigid_polygons_file = args.map[0]
rigid_body = RigidBody(f, ax, rigid_polygons_file)

start = np.array([args.start[0], args.start[1], args.start[2]])
goal = np.array([args.goal[0], args.goal[1], args.goal[2]])

P = 0
N = 1000

print("Start: ", start)
print("Goal: ", goal)
print("Map File: ", rigid_polygons_file)

rrt = RRT(start, goal, rigid_body)

#Finished Creating RRT
while P < N:
    configuration = rigid_body.sample_configuration_collision_free(1)[0]    
    added_successfully = rrt.add_vertex(configuration)
    
    P = P + 1 if added_successfully else P
    
    if P % 100 == 0:
        print(f"FINISHED P = {P}")

path = rrt.generate_path()

print("Path:", path)

#Generate Animation
if len(path) > 0:
    rrt.animation = FuncAnimation(f, rrt.update_animation_configuration, frames = range(0, path.size), blit = True, interval = 800)
plt.show()