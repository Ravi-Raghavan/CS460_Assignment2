import argparse
from matplotlib import pyplot as plt
from utils_p1 import compute_distance, compute_norm, normalize_angle
import numpy as np
from matplotlib.animation import FuncAnimation
from arm_2 import linear_search
import utils_p1 as utl

class Graph:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.nodes = [start]
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)
    
    def add_edge(self, edge):
        self.edges.append(edge)

# retrieve path from graph generated by RRT
def RRT_path(graph, goal):
    pass

# checks if we can sample goal
def sample_goal():
    # Generate a random number between 0 and 1
    random_number = np.random.uniform(0,1)
    # If the random number is less than or equal to 0.05 (5% chance), sample the goal node
    if random_number <= 0.05:
        return True
    else:
        return False
    
 # returns True if arm is collision free
def collision(arm):
    collision = True
    for polygon in arm.polygons:
            #collision check 
            if utl.collides_optimized(arm.points,polygon):
                collision = False
                break
            for joint in arm.joint_boxes:
                if utl.collides_optimized(joint, polygon):
                    collision = False
                    break
    return collision   

#returns arm configuration between 2 given arm configurations
def new_configuration(arm, q_near, q_rand):
    theta1 = q_rand[0]-q_near[0]
    theta2 = q_rand[1]-q_near[1]

    # computs half distance from nearest node to random node
    theta1_dist, theta2_dist = compute_distance(q_near, q_rand)
    distance = compute_norm((theta1_dist, theta2_dist))

    if distance > 0.3:
        theta1_dist = theta1_dist/2
        theta2_dist = theta2_dist/2
        # checks what direction to move in for joint 1
        if theta1 >= 0:
            new_theta1 = q_near[0] + theta1_dist
        else:
            new_theta1 = q_near[0] - theta1_dist
        
        # checks what direction to move in for joint2
        if theta2 >= 0:
            new_theta2 = q_near[1] + theta2_dist
        else:
            new_theta2 = q_near[1] - theta1_dist

        # check for collision free path
        step1 = theta1_dist/5
        step2 = theta2_dist/5
        theta1_temp = q_near[0]
        theta2_temp = q_near[1]

        for _ in range(5):
            if theta1 >= 0:
                theta1_temp = theta1_temp + step1
            else:
                theta1_temp = theta1_temp - step1
            if theta2 >= 0:
                theta2_temp = theta2_temp + step2
            else:
                theta2_temp = theta2_temp - step2

            arm.update_joints((theta1_temp, theta2_temp))
            no_collision = collision(arm)
            if not no_collision:
                return None
        return [new_theta1, new_theta2]
    else: # sample is close enough to nearest node
        arm.update_joints((q_rand[0], q_rand[1]))
        collision_free = collision(arm)
        if not collision_free:
            return None
    return [q_rand[0], q_rand[1]]

# return node if moving distance d to config is collision free
def extend(arm, q_near, q_rand):
    q_new = new_configuration(arm, q_near, q_rand)
    if q_new is not None:
        arm.update_joints(q_new)
        return q_new
    return None

def RRT(arm, start, goal):
    # initialize tree 
    G = Graph(start, goal)

    for _ in range(1,1000):
        if not sample_goal(): # sample a node 
            q_rand = [np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi)]
        else: # sample goal 5% of time
            q_rand = goal
            print(q_rand)

        # find nearest node in tree to q_rand
        q_near = np.array(linear_search(q_rand, 1, G.nodes)).flatten()
    
        # extend nearest node to random node by distance R
        q_new = extend(arm, q_near, q_rand)
        if q_new:
            G.add_node(q_new)
            G.add_edge((q_near, q_new))
            if q_new == goal:
                #path = RRT_path()
                path = True
                return path
    # RRT was unsuccesful 
    return None

# animation of solution path 
def move(arm, path):
    num_frames = len(path) + 1
    index = 0
    # create animation
    def update(frame):
        utl.plt.cla()
        joint1 = path[index][0]
        joint2 = path[index][1]

        if frame == 0:  # start configuration
            arm.plot('lightblue')
        elif frame == 21:  # goal configuration
            arm.plot('thistle')
        else:  # other configurations
            arm.plot('slategrey')

        arm.update_joints((joint1, joint2))
        index += 1
    animation = FuncAnimation(arm.fig, update(path[index]), frames=num_frames, repeat=False)
    utl.plt.show()
    
def main():
    # parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Interpolation along straight line in C-space')
    parser.add_argument('--start', nargs=2, type=float, help='start configuration')
    parser.add_argument('--goal', nargs=2, type=float, help='goal configuration')
    parser.add_argument('--map', type=str, help='polygonal map')
    args = parser.parse_args()

    start = args.start
    goal = args.goal
    map = args.map
    links = [0.3,0.15]

    # initialize robot at starting orientation
    arm = utl.RobotArm(map, links, [start[0], start[1]], joint_radius=0.05, link_width=0.1)
    path = RRT(arm, start, goal)
    print(path)

    # check if a path was found
   # if path:
   #     move(arm, path)
   # else:
   #     print("Unable to find path within 1000 nodes")

if __name__ == "__main__":
    main()