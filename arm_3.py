import argparse
import numpy as np
from matplotlib import pyplot as plt
import utils_p1 as utl

#  robot arm moves from start to goal configuration
def move(arm, goal):
    # keeps moving robot until it reaches goal
    while arm.joint_angles != goal:
        joint1 = arm.joint_angles[0]
        joint2= arm.joint_angles[1]

        if joint1 != goal[0]:
            joint1 = joint1 + np.pi/20

        if joint2 != goal[1]:
            joint2 = joint2 + np.pi/20

        arm.update_joints((joint1,joint2))

        # create animation 
        arm.plot()


def main():
    # parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Interpolation along straight line in C-space')
    parser.add_argument('--start', nargs=2, type=float, help='start configuration')
    parser.add_argument('--goal', nargs=2, type=float, help='goal configuration')
    args = parser.parse_args()

    # store arguments
    start = args.start
    goal = args.goal

    # initialize robot at starting orientation
    arm = utl.RobotArm(map, [0.3,0.15], [start[0], start[1]], joint_radius=0.05, link_width=0.1)

    # move arm from start to goal
    move(arm, goal)

if __name__ == "__main__":
    main()