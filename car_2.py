## Implement RRT For 3.3
from utils_p3 import *
import argparse
from matplotlib.animation import FuncAnimation
import time

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

parser = argparse.ArgumentParser(description="Receive Command Line Arguments for RRT Problem")
parser.add_argument("--start", nargs = 3, type = float, help = "Start Configuration")
parser.add_argument("--goal", nargs = 3, type = float, help = "Goal Configuration")
parser.add_argument("--map", nargs = 1, type = str, help = "Map of Rigid Polygons")
args = parser.parse_args()

start = np.array([args.start[0], args.start[1], args.start[2]])
goal = np.array([args.goal[0], args.goal[1], args.goal[2]])
rigid_polygons_file = args.map[0]

#Define our car 
car = Car(f, ax, rigid_polygons_file, False, start, enable_keyboard_control = False)

#Define our RRT
rrt = RRT(start, goal, car, b = 1, num_integrations = 250)

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

#Compute Path Distance
def path_distance(rrt: RRT, path, controls, integration_steps):
    q = start
    
    L = 0.2
    path_distance = 0
    
    for control_index, control in enumerate(controls):
        u = control.flatten()
        for step in range(integration_steps[control_index]):
            v, phi = u[0], u[1]
            theta = q[2]
            derivative = np.array([v * np.cos(theta), v * np.sin(theta), v * np.tan(phi) / L])
            change_q = (rrt.rigid_body.delta_t) * derivative
            path_distance += rrt.D(change_q)
            q += change_q
    
    return path_distance

P = 0
reached_goal_region = False
goal_index = None

probability = 0.05
iterations = 1

max_iterations = 1000

#Start the timer
start_time = time.time()

while not reached_goal_region and iterations <= max_iterations:
    #Sample Configuration, Pick Goal node with 'probability' Probability, and add Node to RRT
    configuration = car.sample_configuration_collision_free(1)[0]    
    uniform_sampled_number = np.random.uniform(0, 1)
    
    configuration = goal if uniform_sampled_number < probability else configuration
    added_successfully, configuration_index = rrt.add_vertex(configuration)
    
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

#Finished Building RRT
end_time = time.time()
rrt_build_time = end_time - start_time

if not reached_goal_region:    
    #RRT Vertices
    rrt_vertices = rrt.vertices
    
    min_distance = np.inf
    min_translational_distance = np.inf
    min_rotational_distance = np.inf
    
    closest_vertex = None
    for vertex_index, vertex in enumerate(rrt_vertices):
        v = vertex.flatten()
        dt = Dt(v, goal)
        dr = Dr(v, goal)
        d = D(v, goal, 0.7)
        
        if d < min_distance:
            min_distance = d
            min_translational_distance = dt
            min_rotational_distance = dr
            closest_vertex = vertex
            goal_index = vertex_index
    
    
#Generate Path
start_time = time.time()
path, controls, integration_steps = rrt.generate_path(goal_index)
end_time = time.time()

rrt_path_generation_time = end_time - start_time

print(f"Path: {path}")
print(f"Controls: {controls}")
print(f"Integration Steps: {integration_steps}")

print(f"Path Length: {len(path)}")
print(f"Number of Controls: {len(controls)}")
print(f"Length of Integration Steps Array: {len(integration_steps)}")

total_integration_steps = np.sum(integration_steps)
print(f"Total Number of Integration Steps: {total_integration_steps}")

print(f"Reached Goal Region? {reached_goal_region}")
print(f"Path Distance: {path_distance(rrt, path, controls, integration_steps)}")
print(f"RRT Build Time: {rrt_build_time}")
print(f"RRT Path Generation Time: {rrt_path_generation_time}")

    
rrt.animation = FuncAnimation(f, rrt.update_animation_configuration, frames = range(0, total_integration_steps + 1), init_func = rrt.init_animation_configuration, blit = True, interval = 20, repeat = False)
plt.show()