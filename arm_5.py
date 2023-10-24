import argparse
from matplotlib import pyplot as plt
import utils_p1 as utl

# PRM IMPLEMENTATION
def prm(start, goal):

# PRM* IMPLEMENTATION
def prm_star(start,goal):

def main():
    # parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Interpolation along straight line in C-space')
    parser.add_argument('--start', nargs=2, type=float, help='start configuration')
    parser.add_argument('--goal', nargs=2, type=float, help='goal configuration')
    parser.add_argument('--map', nargs=2, type=float, help='polygonal map')
    args = parser.parse_args()

    start = args.start
    goal = args.goal
    map = args.map

    # initialize robot at starting orientation
    arm = utl.RobotArm(map, [0.3,0.15], [start[0], start[1]], joint_radius=0.05, link_width=0.1)
    prm(arm, goal)

if __name__ == "__main__":
    main()