import argparse
import numpy as np
from matplotlib.animation import FuncAnimation
from utils_p1 import compute_distance
import utils_p1 as utl
    
# moves arm from start to goal configuration
def move(arm, start,goal):
    # calculate step size
    distance1 = utl.angle_difference(start[0],goal[0])
    distance2 = utl.angle_difference(start[1],goal[1])
    step1 = distance1/20
    step2 = distance2/20
    configurations1 = []
    configurations2 = []

    theta1 = goal[0]-start[0]
    theta2 = goal[1]-start[1]

    for i in range(22):
        if theta1 >= 0:
            joint1 = start[0] + i * step1
        else:
            joint1 = start[0] - i * step1
        if theta2 >= 0:
            joint2 = start[1] + i * step2
        else:
            joint2 = start[1] -i * step2
        configurations1.append(joint1)
        configurations2.append(joint2)

    # create animation
    def update(frame):
        utl.plt.cla()
        joint1 = configurations1[frame]
        joint2 = configurations2[frame]

        if frame == 0:  # start configuration
            arm.plot('lightblue')
        elif frame == 21:  # goal configuration
            arm.plot('thistle')
        else:  # other configurations
            arm.plot('slategrey')

        arm.update_joints((joint1, joint2))

    animation = FuncAnimation(arm.fig, update, frames=22, repeat=False)
    utl.plt.show()

def main():
    # parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Interpolation along straight line in C-space')
    parser.add_argument('--start', nargs=2, type=float, help='start configuration')
    parser.add_argument('--goal', nargs=2, type=float, help='goal configuration')
    args = parser.parse_args()

    # store arguments
    start = args.start
    goal = args.goal
    lengths = [0.4,0.25]

    # initialize robot at starting orientation
    arm = utl.RobotArm('None', lengths, [start[0], start[1]], joint_radius=0.05, link_width=0.16611) #link_width is 1, 0.16611 makes it appear correctly

    # move arm from start to goal
    move(arm, start, goal)

if __name__ == "__main__":
    main()