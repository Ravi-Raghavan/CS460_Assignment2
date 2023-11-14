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

if (rigid_body.check_rigid_body_collision(rigid_body.generate_rigid_body_from_configuration(start))):
    print("The start configuration collide with the boundary or with obstacles in the environment. Please try again with different start configuration")
    exit(0)

if (rigid_body.check_rigid_body_collision(rigid_body.generate_rigid_body_from_configuration(goal))):
    print("The goal configuration collide with the boundary or with obstacles in the environment. Please try again with different goal configuration")
    exit(0)

P = 0
N = 1000

print("Start: ", start)
print("Goal: ", goal)
print("Map File: ", rigid_polygons_file)

rrt = RRT(start, goal, rigid_body)

#Calculate Translational Distance between two configurations
def Dt(q1, q2):
    q1 = q1.flatten()
    q2 = q2.flatten()
    
    dt = np.linalg.norm(q1[:-1] - q2[:-1])
    return dt

#Calculate Rotational Distance between two configurations
def Dr(q1, q2):
    q1 = q1.flatten()
    q2 = q2.flatten()
    
    dr = min(abs(q1[2] - q2[2]), 2 * np.pi - abs(q1[2] - q2[2]))
    return dr
    

#Calculate Distance between two Configurations
def D(q1, q2, alpha):
    #calculate dt and dr
    dt = Dt(q1, q2)
    dr = Dr(q1, q2)
    
    return (alpha * dt) + ((1 - alpha) * dr)

#Finished Creating RRT
probability = 0.05
iterations = 1
reached_goal_region = False

while P < N:
    configuration = rigid_body.sample_configuration_collision_free(1)[0] 
    uniform_sampled_number = np.random.uniform(0, 1)
    
    configuration = goal if uniform_sampled_number < probability else configuration   
    added_successfully = rrt.add_vertex(configuration)
    
    #RRT Vertices
    rrt_vertices = rrt.vertices
    
    min_distance = np.inf
    min_translational_distance = np.inf
    min_rotational_distance = np.inf
    
    dt_values = []
    dr_values = []
    
    closest_vertex = None
    for vertex_index, vertex in enumerate(rrt_vertices):
        v = vertex.flatten()
        dt = Dt(v, goal)
        dr = Dr(v, goal)
        d = D(v, goal, 0.7)
        
        if dt < 0.1 and dr < 0.5:
            reached_goal_region = True
            if d < min_distance:
                min_distance = d
                min_translational_distance = dt
                min_rotational_distance = dr
                closest_vertex = vertex
                goal_index = vertex_index
        
        dt_values.append(dt)
        dr_values.append(dr)
    
    P = P + 1 if added_successfully else P
    
    if P % 10 == 0:
        print(f"P = {P}, Dt = {np.min(dt_values)}, Dr = {np.min(dr_values)}, minDT = {min_translational_distance}, minDR = {min_rotational_distance}, In Goal Region: {reached_goal_region}, Uniform Distribution Number: {uniform_sampled_number}, Probability: {probability}")
    
    iterations = iterations + 1

path, configuration_derivatives, timestep_array = rrt.generate_path()

print("Path:", path)
print("Configuration Derivatives: ", configuration_derivatives)
print("Timestep Array: ", timestep_array)

#Generate Animation
if len(path) > 0:
    total_timesteps = np.sum(timestep_array)
    print("Total Timesteps: ", total_timesteps)
    rrt.animation = FuncAnimation(f, rrt.update_animation_configuration, frames = range(0, total_timesteps + 1), init_func = rrt.init_animation_configuration, blit = True, interval = 30, repeat = False)
    plt.show()