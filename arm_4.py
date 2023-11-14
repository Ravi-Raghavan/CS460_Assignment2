import argparse
from matplotlib import pyplot as plt
from utils_p1 import compute_distance, compute_norm, linear_search, Graph, normalize_angle, wrap
import numpy as np
import matplotlib.animation as animation
import utils_p1 as utl

# Checks if goal can be sampled (5% of the time)
def sample_goal():
    random_number = np.random.uniform(0,1)
    if random_number <= 0.05:
        return True
    else:
        return False
    
# Checks if the arm is in a collision-free configuration
def collision_free(arm):
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

# Returns a new arm configuration between 2 given arm configurations (if possible)
def new_configuration(arm, q_near, q_rand):
    theta1 = q_rand[0]-q_near[0]
    theta2 = q_rand[1]-q_near[1]

    # Compute distance between nearest config and sample config
    theta1_dist, theta2_dist = compute_distance(q_near, q_rand)
    distance = compute_norm((theta1_dist, theta2_dist))

    # If distance is less than (pi/20,pi/20) away, no need to sample in between
    if distance > 0.2:
        theta1_dist = theta1_dist/2
        theta2_dist = theta2_dist/2

        # Distance to move theta1
        if theta1 >= 0:
            new_theta1 = q_near[0] + theta1_dist
        else:
            new_theta1 = q_near[0] - theta1_dist

        # Distance to move theta2
        if theta2 >= 0:
            new_theta2 = q_near[1] + theta2_dist
        else:
            new_theta2 = q_near[1] - theta1_dist

        # Check for collision-free path
        step1 = theta1_dist/5
        step2 = theta2_dist/5
        theta1_temp = q_near[0]
        theta2_temp = q_near[1]

        # Sample along path from current config to new config
        for _ in range(5):
            # Direction to move theta1
            if theta1 >= 0:
                theta1_temp = theta1_temp + step1
            else:
                theta1_temp = theta1_temp - step1
            # Direction to move theta2
            if theta2 >= 0:
                theta2_temp = theta2_temp + step2
            else:
                theta2_temp = theta2_temp - step2

            # Check arm for collisions
            arm.update_joints((theta1_temp, theta2_temp))
            no_collision = collision_free(arm)
            if not no_collision:
                return None
        return [new_theta1, new_theta2]
    
    else:
        arm.update_joints((q_rand[0], q_rand[1]))
        no_collision = collision_free(arm)
        if not no_collision:
            return None
    return [q_rand[0], q_rand[1]]

# Return a new config between the nearest and sample configuration (if possible)
def extend(arm, q_near, q_rand):
    q_new = new_configuration(arm, q_near, q_rand)
    if q_new is not None:
        arm.update_joints(q_new)
        return q_new
    return None

# RRT Implementation
def RRT(arm, start, goal):
    V = [start]
    E = []
    graph = Graph()
    graph.add(tuple(start),'') 
    num_nodes = 0
    
    while num_nodes <= 1000:
        # Check if goal can be sampled
        if not sample_goal(): 
            q_rand = [np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi)]
        else: 
            q_rand = goal

        # Find nearest configuration to sample
        q_near = np.array(linear_search(q_rand, 1, V)).flatten()
    
        # Extend nearest cofiguration in direction of sample
        q_new = extend(arm, q_near, q_rand)

        # Add new configuration to graph
        if q_new:
            V.append(q_new)
            E.append((q_near,q_new))
            num_nodes += 1
            graph.add(tuple(q_near),tuple(q_new))
            # check if goal was found
            if q_new[0] == goal[0] and q_new[1] == goal[1]:
                path = graph.shortest_path(tuple(start), tuple(goal))
                animation = cspace(E,start)
                return (path,animation)
    # RRT was unsuccesful after 1000 configurations was sampled
    animation = cspace(E,start) 
    return (None,animation)

# Animation of solution path in workspace
def move(arm, path):
    animations = []
    num_frames = len(path)
    # create animation
    def update(index):
        utl.plt.cla()
        arm.ax.clear()
        arm.plot_polygons()
        arm.update_joints(path[index])
        if index == 0:  # start configuration
            arm.plot('lightblue')
        elif index == len(path)-1:  # goal configuration
            arm.plot('thistle')
        else:  # other configurations
            arm.plot('slategrey')
        index += 1
    ani = animation.FuncAnimation(arm.fig, update, frames=num_frames, interval = 500, repeat=False)
    animations.append(ani)
    return animations

# Animation of Configuration Space 
def cspace(edges,start):
    # Initialize the plot
    fig, ax = plt.subplots()
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 2*np.pi)
    start1 = start[0]
    start2 = start[1]
    animations = []

    # Function to update the plot during animation
    def update(num):
        ax.clear()
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        plt.plot(normalize_angle(start1), normalize_angle(start2), marker='o', linestyle='', color ='steelblue')
        # Plot the edges
        current_edges = edges[:num]
        for edge in current_edges:
            start = (normalize_angle(edge[0][0]), normalize_angle(edge[0][1]))
            goal = (normalize_angle(edge[1][0]),normalize_angle(edge[1][1]))

            # Detect if wrap around occurs
            theta1, direction1 = wrap(start[0],goal[0])
            theta2, direction2 = wrap(start[1],goal[1])

            # Check where wrapping occurs
            if theta1 and theta2: # BOTH AXES WRAP
               # same direction wrapping
                if direction1 == direction2:
                    if direction1 == '2pi': 
                        ax.plot([start[0], start[0]+(2*np.pi-start[0]+goal[0])], [start[1], start[1]+(2*np.pi-start[1]+goal[1])], 'skyblue')
                        ax.plot([goal[0], goal[0]-(2*np.pi-start[0]+goal[0])], [goal[1], goal[1]-(2*np.pi-start[1]+goal[1])], 'skyblue')
                    else:
                        ax.plot([start[0], start[0]-(2*np.pi-goal[0]+start[0])], [start[1], start[1]-(2*np.pi-goal[1]+start[1])], 'skyblue')
                        ax.plot([goal[0], goal[0]+(2*np.pi-goal[0]+start[0])], [goal[1], goal[1]-(2*np.pi-goal[1]+start[1])], 'skyblue')
               # theta 1: 0 to 2pi AND theta 2: 2pi to 0
                elif direction1 == '0':
                    ax.plot([start[0], start[0]-(start[0]+(2*np.pi-goal[0]))], [start[1], start[1]+(goal[1]+(2*np.pi-start[1]))], 'skyblue')
                    ax.plot([goal[0], goal[0]+start[0]+(2*np.pi-goal[0])], [goal[1], goal[1]-(goal[1]+2*np.pi-start[1])], 'skyblue')
                # theta 1: 2pi to 0 AND theta 2: 0 to 2pi
                elif direction1 == '2pi':
                    ax.plot([start[0], start[0]+(2*np.pi-start[0])+goal[0]], [start[1], start[1]-(start[1]+(2*np.pi-goal[1]))], 'skyblue')
                    ax.plot([goal[0], goal[0]-(goal[0]+(2*np.pi-start[0]))], [goal[1], goal[1]+(start[1]+2*np.pi-goal[1])], 'skyblue')
            # THETA 1 WRAPS ONLY
            elif theta1:
                # theta1: 0 to 2pi
                if direction1 == '0': #PROBLEMMMMM
                    ax.plot([start[0], start[0]-(2*np.pi-goal[0]+start[0])], [start[1], goal[1]], 'skyblue')
                    ax.plot([goal[0], goal[0]+(2*np.pi-goal[0]+start[0])], [goal[1], start[1]], 'skyblue')
                # theta1: 2pi to 0
                else:
                    ax.plot([start[0], start[0]+(2*np.pi-start[0]+goal[0])], [start[1],goal[1]], 'skyblue')
                    ax.plot([goal[0], goal[0]-(goal[0]+2*np.pi-start[0])], [goal[1], start[1]], 'skyblue')
            # THETA 2 WRAPS ONLY
            elif theta2:
                # theta2: 0 to 2pi
                if direction2 == '0':
                    ax.plot([start[0], goal[0]], [start[1], start[1]-(start[1]+2*np.pi-goal[1])], 'skyblue')
                    ax.plot([goal[0], start[0]], [goal[1], goal[1]+(2*np.pi-goal[1])+start[1]], 'skyblue')
                # theta2: 2pi to 0
                else:
                    ax.plot([start[0], goal[0]], [start[1], start[1]+goal[1]+(2*np.pi-start[1])], 'skyblue')
                    ax.plot([goal[0], start[0]], [goal[1], goal[1]-(goal[1]+2*np.pi-start[1])], 'skyblue')
            else: # No wrapping on either axis
                ax.plot([start[0], goal[0]], [start[1], goal[1]], 'skyblue')

            # Plot the vertices
            plt.scatter(start[0],start[1], color = 'skyblue',marker='o')
            plt.scatter(goal[0],goal[1], color = 'skyblue',marker='o')
            plt.plot(normalize_angle(start1), normalize_angle(start2), marker='o', linestyle='', color ='steelblue')

    # Animate the plot
    ani = animation.FuncAnimation(fig, update, frames=len(edges), repeat=False)
    animations.append(ani)
    return animations
    
def main():
    # Parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Interpolation along straight line in C-space')
    parser.add_argument('--start', nargs=2, type=float, help='start configuration')
    parser.add_argument('--goal', nargs=2, type=float, help='goal configuration')
    parser.add_argument('--map', type=str, help='polygonal map')
    args = parser.parse_args()
    start = args.start
    goal = args.goal
    map = args.map
    links = [0.4,0.25]

    # Initialize robot
    arm = utl.RobotArm(map, links, [start[0], start[1]], joint_radius=0.05, link_width=0.16611) #link_width is 1, 0.16611 makes it appear correctly
    arm.update_joints(goal)
    if not collision_free(arm):
        print("Goal is not collision-free")
        exit()

    # RRT Implementation
    path, animation1 = RRT(arm, start, goal)

    # Check if a path was found
    if path:
       animation2 = move(arm, path)
    else:
        print("Unable to find path within 1000 nodes")
    utl.plt.show()

if __name__ == "__main__":
    main()