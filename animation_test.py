from utils_p2 import RigidBody, RRT

import argparse
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

# Create an initial empty plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0, 2)  # Set x-axis limits
ax.set_ylim(0, 2)  # Set y-axis limits
ax.set_zlim(-np.pi, np.pi)  # Set z-axis limits

#Parse Arguments
parser = argparse.ArgumentParser(description="Receive Command Line Arguments for RRT Problem")
parser.add_argument("--start", nargs = 3, type = float, help = "Start Configuration")
parser.add_argument("--goal", nargs = 3, type = float, help = "Goal Configuration")
args = parser.parse_args()

start = np.array([args.start[0], args.start[1], args.start[2]])
goal = np.array([args.goal[0], args.goal[1], args.goal[2]])

P = 0
N = 1000

print("Start: ", start)
print("Goal: ", goal)

f2,ax2 = plt.subplots(dpi = 100)
ax2.set_aspect("equal")

rigid_body =  RigidBody(f2, ax2, None)
rrt = RRT(start, goal, rigid_body, fig, ax)

# Function to update the plot at each frame
def update(frame):
    print("Frame:", frame)
    
    configuration = rigid_body.sample_configuration_collision_free(1)[0]    
    rrt.add_3D_animation_vertex(configuration)
    return rrt.update_3D_animation_configuration(frame)


# Create the animation
ani = FuncAnimation(fig, update, frames = 10, blit = True, interval = 800)

plt.show()
