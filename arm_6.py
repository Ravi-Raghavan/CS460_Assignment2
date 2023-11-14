import argparse
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
import utils_p1 as utl
from utils_p1 import linear_search, PRM_Graph, normalize_angle, wrap, compute_distance, compute_norm

# Checks if configuration is collision free
def free(arm):
    no_collision = True
    for polygon in arm.polygons:
            #collision check 
            if utl.collides_optimized(arm.points,polygon):
                no_collision = False
                break
            for joint in arm.joint_boxes:
                if utl.collides_optimized(joint, polygon):
                    no_collision = False
                    break
    return no_collision

# Checks if arm is able to move from one config to another w/o collisions
def visible(arm, vertex, neighbor):
    theta1 = neighbor[0]-vertex[0]
    theta2 = neighbor[1]-vertex[1]

    # Compute distance between nearest config and sample config
    theta1_dist, theta2_dist = compute_distance(vertex, neighbor)
    distance = compute_norm((theta1_dist, theta2_dist))

    # If distance is less than (pi/20,pi/20) away, no need to sample in between
    if distance > 0.2:
        # Check for collision-free path
        step1 = theta1_dist/5
        step2 = theta2_dist/5
        theta1_temp = vertex[0]
        theta2_temp = vertex[1]

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
            no_collision = free(arm)
            if not no_collision:
                return (False, None)
        return (True, distance)
    
    else:
        arm.update_joints((neighbor[0], neighbor[1]))
        no_collision = free(arm)
        if not no_collision:
            return (False, None)
    return (True, distance)

# Return vertices and edges roadmap 
def roadmap(arm, N, start, goal, k): 
    V = []
    E = []
    graph = PRM_Graph()

    # Sample N random configurations
    q_rand = (np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi))
    for _ in range(N):
        q_rand = (np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi))
        while not free(arm):
            q_rand = (np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi))
        V.append(q_rand)

    # Connect nodes to its k-nearest neighbors
    for vertex in V:
        neighbors = linear_search(vertex, k, V)
        for neighbor in neighbors:
            isVisible, distance = visible(arm, goal, neighbor)
            if isVisible:
                E.append((vertex,neighbor))
                graph.add(tuple(vertex),tuple(neighbor),distance)

    #find nearest neighbors of start and goal configurations to connect to graph
    q_start_neighbor = linear_search(start, k, V)
    q_goal_neighbor = linear_search(goal, k, V)

    # Connect start to nearest neighbors
    for neighbor in q_start_neighbor:
            isVisible, distance = visible(arm, goal, neighbor)
            if isVisible:
                E.insert(0,(start, neighbor))
                graph.add(tuple(start),tuple(neighbor),distance)
    
    for neighbor in q_goal_neighbor:
            isVisible, distance = visible(arm, goal, neighbor)
            if isVisible:
                E.insert(0,(goal,neighbor))
                graph.add(tuple(goal),tuple(neighbor),distance)

    # Animation of Configuration Space
    animation = cspace(V,E,start,goal)
    return (graph, animation)

# Returns a path to the goal
def query(graph, start, goal):
    # Search roadmap to determine if a path exists using dijkstras
    path = graph.dijkstras(tuple(start),tuple(goal))
    if path:
        return path
    else:
        return None

# PRM* IMPLEMENTATION -> k = log(n)
def prm_star(arm, start, goal):
    N = 1000
    k = np.log(N)
    k = k.astype(int)
    new_roadmap, cspace_animation = roadmap(arm, N, start, goal, k)
    path = query(new_roadmap, start, goal)

    # Animate solution path in workspace
    if path:
        workspace_animation = move(arm, path)
    else:
        print("Unable to find feasible path after 1000 nodes")
        workspace_animation = None
    return (cspace_animation, workspace_animation)

# Animates Configuration Space
def cspace(vertices, edges, start, goal): 
    if len(edges) != 0:
        # Initialize the plot
        fig, ax = plt.subplots()
        plt.xlim(0, 2*np.pi)
        plt.ylim(0, 2*np.pi)
        animations = []
        # plot vertices
        x,y = zip(*vertices)
        waypoint1 = [normalize_angle(start[0]),normalize_angle(goal[0])]
        waypoint2 = [normalize_angle(start[1]),normalize_angle(goal[1])]

        # Function to update the plot during animation
        def update(num):
            ax.clear()
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, 2*np.pi)
            plt.scatter(x,y, marker='o', color = 'olivedrab')
            plt.plot(waypoint1, waypoint2, marker='o', linestyle='', color ='firebrick')
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
                            ax.plot([start[0], start[0]+(2*np.pi-start[0]+goal[0])], [start[1], start[1]+(2*np.pi-start[1]+goal[1])], 'olivedrab')
                            ax.plot([goal[0], goal[0]-(2*np.pi-start[0]+goal[0])], [goal[1], goal[1]-(2*np.pi-start[1]+goal[1])], 'olivedrab')
                        else:
                            ax.plot([start[0], start[0]-(2*np.pi-goal[0]+start[0])], [start[1], start[1]-(2*np.pi-goal[1]+start[1])], 'olivedrab')
                            ax.plot([goal[0], goal[0]+(2*np.pi-goal[0]+start[0])], [goal[1], goal[1]-(2*np.pi-goal[1]+start[1])], 'olivedrab')
                # theta 1: 0 to 2pi AND theta 2: 2pi to 0
                    elif direction1 == '0':
                        ax.plot([start[0], start[0]-(start[0]+(2*np.pi-goal[0]))], [start[1], start[1]+(goal[1]+(2*np.pi-start[1]))], 'olivedrab')
                        ax.plot([goal[0], goal[0]+start[0]+(2*np.pi-goal[0])], [goal[1], goal[1]-(goal[1]+2*np.pi-start[1])], 'olivedrab')
                    # theta 1: 2pi to 0 AND theta 2: 0 to 2pi
                    elif direction1 == '2pi':
                        ax.plot([start[0], start[0]+(2*np.pi-start[0])+goal[0]], [start[1], start[1]-(start[1]+(2*np.pi-goal[1]))], 'olivedrab')
                        ax.plot([goal[0], goal[0]-(goal[0]+(2*np.pi-start[0]))], [goal[1], goal[1]+(start[1]+2*np.pi-goal[1])], 'olivedrab')
                # THETA 1 WRAPS ONLY
                elif theta1:
                    # theta1: 0 to 2pi
                    if direction1 == '0': #PROBLEMMMMM
                        ax.plot([start[0], start[0]-(2*np.pi-goal[0]+start[0])], [start[1], goal[1]], 'olivedrab')
                        ax.plot([goal[0], goal[0]+(2*np.pi-goal[0]+start[0])], [goal[1], start[1]], 'olivedrab')
                    # theta1: 2pi to 0
                    else:
                        ax.plot([start[0], start[0]+(2*np.pi-start[0]+goal[0])], [start[1],goal[1]], 'olivedrab')
                        ax.plot([goal[0], goal[0]-(goal[0]+2*np.pi-start[0])], [goal[1], start[1]], 'olivedrab')
                # THETA 2 WRAPS ONLY
                elif theta2:
                    # theta2: 0 to 2pi
                    if direction2 == '0':
                        ax.plot([start[0], goal[0]], [start[1], start[1]-(start[1]+2*np.pi-goal[1])], 'olivedrab')
                        ax.plot([goal[0], start[0]], [goal[1], goal[1]+(2*np.pi-goal[1])+start[1]], 'olivedrab')
                    # theta2: 2pi to 0
                    else:
                        ax.plot([start[0], goal[0]], [start[1], start[1]+goal[1]+(2*np.pi-start[1])], 'olivedrab')
                        ax.plot([goal[0], start[0]], [goal[1], goal[1]-(goal[1]+2*np.pi-start[1])], 'olivedrab')
                else: # No wrapping on either axis
                    ax.plot([start[0], goal[0]], [start[1], goal[1]], 'olivedrab')
                plt.plot(waypoint1, waypoint2, marker='o', linestyle='', color ='firebrick')

        # Animate the plot
        ani = animation.FuncAnimation(fig, update, frames=len(edges), repeat=False)
        animations.append(ani)
        return animations
    else:
        print("NO EDGES")

# Animates solution path of arm in workspace
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
    ani = animation.FuncAnimation(arm.fig, update, frames=num_frames, interval= 500, repeat=False)
    animations.append(ani)
    return animations

def main():
    # Parse command line arguments for arguments
    parser = argparse.ArgumentParser(description='Interpolation along straight line in C-space')
    parser.add_argument('--start', nargs=2, type=float, help='start configuration')
    parser.add_argument('--goal', nargs=2, type=float, help='goal configuration')
    parser.add_argument('--map',type=str, help='polygonal map')
    args = parser.parse_args()
    start = args.start
    goal = args.goal
    map = args.map

    # Initialize robot at starting orientation
    arm = utl.RobotArm(map, [0.4,0.25], [start[0], start[1]], joint_radius=0.05, link_width=0.16611) #link_width is 1, 0.16611 makes it appear correctly
    cspace_animation, workspace_animation = prm_star(arm, start, goal)
    utl.plt.show()

if __name__ == "__main__":
    main()