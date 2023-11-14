### Problem 2: Motion Planning for a Rigid Body
## 2.3: Interpolation along the straight line in the C-space
from utils_p2 import RigidBody

import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

#Set up Plot Information
f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

#Load in map of rigid polygons
parser = argparse.ArgumentParser(description="Receive Command Line Arguments for Interpolation Along Straight Line Problem")
parser.add_argument("--start", nargs = 3, type = float, help = "Start Configuration")
parser.add_argument("--goal", nargs = 3, type = float, help = "Goal Configuration")
args = parser.parse_args()

start = np.array([args.start[0], args.start[1], args.start[2]])
goal = np.array([args.goal[0], args.goal[1], args.goal[2]])
timesteps = 25

rigid_body = RigidBody(f, ax, None, start, goal, timesteps)

if (rigid_body.check_rigid_body_collision(rigid_body.generate_rigid_body_from_configuration(start))):
    print("The start configuration collide with the boundary or with obstacles in the environment. Please try again with different start configuration")
    exit(0)

if (rigid_body.check_rigid_body_collision(rigid_body.generate_rigid_body_from_configuration(goal))):
    print("The goal configuration collide with the boundary or with obstacles in the environment. Please try again with different goal configuration")
    exit(0)

ani = FuncAnimation(f, rigid_body.update_animation_configuration, frames=range(0, timesteps + 1), init_func = rigid_body.init_animation_configuration,  blit = True, interval = 800, repeat = False)
plt.show()