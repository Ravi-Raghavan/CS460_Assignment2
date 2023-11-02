import argparse
from matplotlib.animation import FuncAnimation
import utils_p1 as utl
    
# moves arm from start to goal configuration
def move(arm, start,goal):
    # calculate step size
    distance1 = utl.angle_difference(start[0],goal[0])
    distance2 = utl.angle_difference(start[1],goal[1])
    step1 = distance1/20
    step2 = distance2/20

    # create animation
    def update(frame):
        utl.plt.cla()
        joint1 = start[0] + frame * step1
        joint2 = start[1] + frame * step2

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

    # initialize robot at starting orientation
    arm = utl.RobotArm('None', [0.3,0.15], [start[0], start[1]], joint_radius=0.05, link_width=0.1)

    # move arm from start to goal
    move(arm, start, goal)

if __name__ == "__main__":
    main()