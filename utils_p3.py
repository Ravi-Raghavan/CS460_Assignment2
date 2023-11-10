### In this file, I defined utility functions for collision checking, Rigid Body for Problem 3
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

#given a polygon as numpy array, get its edges as an array
#CONFIRMED WORKS
#Expected Format of Input: "polygon" is an (n + 1) x 2 array. There are n vertices in the polygon but the 1st vertex has to be repeated at the end to indicate polygon is closed
#Returns: list of form [edge 1, edge 2, ...., edge n] where edge i is of the form [vertex i, vertex i + 1]
def get_edges(polygon):
    V = polygon.shape[0] - 1 #Get number of vertices in numpy array. subtract 1 because first and last vertex is repeated
    edges = [[polygon[i], polygon[i + 1]] for i in range(V)]
    return edges

#Check for collision between two bounding boxes
#returns True if a collision has been detected, else returns False
#CONFIRMED WORKS
def check_bounding_box_collision(bbox1, bbox2):
    return not (bbox1[1][0] < bbox2[0][0] or 
                bbox1[0][0] > bbox2[1][0] or 
                bbox1[1][1] < bbox2[0][1] or 
                bbox1[0][1] > bbox2[1][1])


#returns True if a collision has been detected, else returns False
#Note: both polygons are an (n + 1) x 2 array. There are n vertices in the polygon but the 1st vertex has to be repeated at the end to indicate polygon is closed
#CONFIRMED WORKS
def SAT(poly1, poly2):
    poly1_edges = get_edges(poly1)
    poly2_edges = get_edges(poly2)
    edges = poly1_edges + poly2_edges
    
    for edge in edges:
        edge_vector = edge[1] - edge[0]
        normal_vector = np.array([-1 * edge_vector[1], edge_vector[0]])
        normal_vector /= np.linalg.norm(normal_vector)
        
        poly1_projections = []
        poly2_projections = []
        
        #Compute Projections
        for vertex in poly1:
            projection = np.dot(vertex, normal_vector) * normal_vector
            poly1_projections.append(projection[0])
        
        for vertex in poly2:
            projection = np.dot(vertex, normal_vector) * normal_vector
            poly2_projections.append(projection[0])
        
        poly1_projections = np.sort(np.array(poly1_projections))
        poly2_projections = np.sort(np.array(poly2_projections))
        
        #If we have found a gap, we know that there is NO collision. Hence, just return False
        if (poly2_projections[-1] < poly1_projections[0] or poly2_projections[0] > poly1_projections[-1]):
            return False
    
    return True

## Optimized Approach for Detecting Polygon Collision
#Note: both polygons are an (n + 1) x 2 array. There are n vertices in the polygon but the 1st vertex has to be repeated at the end to indicate polygon is closed
#CONFIRMED WORKS
#returns True if a collision has been detected, else returns False
def check_polygon_collision(poly1, poly2):
    bounding_boxes = [np.array([np.min(polygon, axis=0), np.max(polygon, axis=0)]) for polygon in [poly1, poly2]]
    if (check_bounding_box_collision(bounding_boxes[0], bounding_boxes[1])):
        return SAT(poly1, poly2)
    
#Python Class for 2D Rigid Body
class Car:
    def __init__(self, f, ax, file, randomize_configuration = True, selected_configuration = None):
        #Store figure and axes as instance variables
        self.f = f
        self.ax = ax
        
        #set axis limits for x and y axes
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        
        #set title for axis
        ax.set_title('Car Simulation', fontsize = 12)
        
        #Load Polygon Data
        self.polygonal_obstacles = np.empty(shape = (0,0))
        if file != None:
            self.polygonal_obstacles = np.load(file, allow_pickle= True)

            #Plot polygons from the npy file
            for index in range(len(self.polygonal_obstacles)):
                self.ax.fill([vertex[0] for vertex in self.polygonal_obstacles[index]], [vertex[1] for vertex in self.polygonal_obstacles[index]], alpha=.25, fc='white', ec='black')
        
        #Set up dimensions for car
        self.W = 0.2
        self.H = 0.1
        self.L = 0.2
        
        #Set up dimensions for wheels
        self.wheel_W = self.L / 4
        self.wheel_H = self.H / 8
        
        #Set up delta_t information
        self.delta_t = 0.01
        
        #Set up control input
        self.control_input = np.zeros(shape = (2,))
        
        #Initialize Configuration and rotation_point
        self.configuration = self.initialize_configuration() if randomize_configuration else selected_configuration
        self.rotation_points = np.empty(shape = (0, 2))
        self.rotation_points = np.vstack((self.rotation_points, self.configuration[:2]))
        print(f"Initial Configuration: {self.configuration} and Rotation Point of {self.rotation_points[0]}")
        
        #Initialize Path
        self.path = Line2D(self.rotation_points[:, 0].flatten(), self.rotation_points[:, 1].flatten())
        ax.add_line(self.path)
        
        #Register Keyboard Handler
        self.f.canvas.mpl_connect('key_press_event', self.keyboard_event_handler)
        
        #Plot Car and Wheels based on initial configuration
        rigid_body = self.generate_rigid_body_from_configuration(self.configuration)
        wheels = self.generate_wheels_from_configuration(self.configuration)
        
        self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='black')
        self.ax.add_patch(self.patch)
        
        self.bottom_left_wheel_patch = matplotlib.patches.Polygon(wheels[0], closed=True, facecolor = 'none', edgecolor='black')
        self.bottom_right_wheel_patch = matplotlib.patches.Polygon(wheels[1], closed=True, facecolor = 'none', edgecolor='black')
        self.top_left_wheel_patch = matplotlib.patches.Polygon(wheels[2], closed=True, facecolor = 'none', edgecolor='black')
        self.top_right_wheel_patch = matplotlib.patches.Polygon(wheels[3], closed=True, facecolor = 'none', edgecolor='black')
        
        self.ax.add_patch(self.bottom_left_wheel_patch)
        self.ax.add_patch(self.bottom_right_wheel_patch)
        self.ax.add_patch(self.top_left_wheel_patch)
        self.ax.add_patch(self.top_right_wheel_patch)
    
    #Set State
    #Assumption: x, y are scalar values
    #theta is a radian value
    def set_state(self, x, y, theta):
        #constrain theta to be between -pi and pi
        theta = theta - (2 * np.pi * (np.floor((theta + np.pi) / (2 * np.pi))))
        self.configuration = np.array([x, y, theta])
    
    #Set Control Input
    #Assumption: v is a scalar value
    #phi is a radian value
    def set_control_input(self, v, phi):
        #Ensure v is within the bounds of [-0.5, 0.5]
        if v < 0:
            v = max(v, -0.5)
        elif v > 0:
            v = min(v, 0.5)
        
        #Ensure phi is between the bounds of [-pi/4, pi/4]
        if phi < 0:
            phi = max(phi, -1 * np.pi/4)
        elif phi > 0:
            phi = min(phi, np.pi/4)
        
        self.control_input = np.array([v, phi])
    
    #Returns True if rigid body is on boundary of discrete grid environment, Else returns False
    def is_rigid_body_on_boundary(self, rigid_body):
        for vertex in rigid_body:
            if vertex[0] <= 0 or vertex[0] >= 2 or vertex[1] <= 0 or vertex[1] >= 2:
                return True
        return False

    #Check to see if the rigid body collides with other polygons in workspace
    #This entails to checking to see if the Rigid Body collides with other polygons in our list or if the rigid boundary is on the boundary
    #Returns True if there is a collision. Else, returns False
    def check_rigid_body_collision(self, rigid_body):
        for polygon in self.polygonal_obstacles:
            if (check_polygon_collision(polygon, np.vstack((rigid_body, rigid_body[0])))):
                return True
        
        return self.is_rigid_body_on_boundary(rigid_body)
    
    #Check to see if the wheels of the car collide with obstacles in workspace
    def check_wheel_collision(self, wheels):
        for wheel in wheels:
            if self.check_rigid_body_collision(wheel):
                return True
        
        return False
    
    #Initialize the configuration of car
    def initialize_configuration(self):
        configuration = np.array([np.random.uniform(0, 2, size = None), np.random.uniform(0, 2, size = None), np.random.uniform(-1 * np.pi, np.pi, size = None)]) #randomly choose a configuration        
        free = False #determines whether the configuration is collision free
        
        while not free:
            rigid_body = self.generate_rigid_body_from_configuration(configuration)
            wheels = self.generate_wheels_from_configuration(configuration)
            
            if (not self.check_rigid_body_collision(rigid_body)) and (not self.check_wheel_collision(wheels)):
                free = True
                break
            
            configuration = np.array([np.random.uniform(0, 2, size = None), np.random.uniform(0, 2, size = None), np.random.uniform(-1 * np.pi, np.pi, size = None)]) #randomly choose a configuration
        
        return configuration
    
    #Given a configuration, get the placement of the wheels
    def generate_wheels_from_configuration(self, configuration):
        x, y = configuration[0], configuration[1]
        theta = configuration[2]
        w, h, L = self.W, self.H, self.L
        phi = self.control_input[1]
        
        #Set up wheel centers with respect to (x, y) being at origin
        bottom_left_wheel_center = np.array([L/2 - w/2, -1 * h/4]) 
        bottom_right_wheel_center = np.array([w/2 + L/2, -1 * h/4])
        top_right_wheel_center = np.array([w/2 + L/2, h/4]) 
        top_left_wheel_center = np.array([L/2 - w/2, h/4])
        
        #Set up rigid body coordinates of each wheel with respect to (x, y) being at origin
        bottom_left_wheel = np.array([bottom_left_wheel_center + np.array([self.wheel_W / 2, -1 * self.wheel_H / 2]),
                                      bottom_left_wheel_center + np.array([self.wheel_W / 2, self.wheel_H / 2]),
                                      bottom_left_wheel_center + np.array([-1 * self.wheel_W / 2, self.wheel_H / 2]),
                                      bottom_left_wheel_center + np.array([-1 * self.wheel_W / 2, -1 * self.wheel_H / 2])])
        
        bottom_right_wheel = np.array([bottom_right_wheel_center + np.array([self.wheel_W / 2, -1 * self.wheel_H / 2]),
                                      bottom_right_wheel_center + np.array([self.wheel_W / 2, self.wheel_H / 2]),
                                      bottom_right_wheel_center + np.array([-1 * self.wheel_W / 2, self.wheel_H / 2]),
                                      bottom_right_wheel_center + np.array([-1 * self.wheel_W / 2, -1 * self.wheel_H / 2])])
        
        top_left_wheel = np.array([top_left_wheel_center + np.array([self.wheel_W / 2, -1 * self.wheel_H / 2]),
                                      top_left_wheel_center + np.array([self.wheel_W / 2, self.wheel_H / 2]),
                                      top_left_wheel_center + np.array([-1 * self.wheel_W / 2, self.wheel_H / 2]),
                                      top_left_wheel_center + np.array([-1 * self.wheel_W / 2, -1 * self.wheel_H / 2])])
        
        top_right_wheel = np.array([top_right_wheel_center + np.array([self.wheel_W / 2, -1 * self.wheel_H / 2]),
                                      top_right_wheel_center + np.array([self.wheel_W / 2, self.wheel_H / 2]),
                                      top_right_wheel_center + np.array([-1 * self.wheel_W / 2, self.wheel_H / 2]),
                                      top_right_wheel_center + np.array([-1 * self.wheel_W / 2, -1 * self.wheel_H / 2])])
        
        #Set up transformation matrix
        transformation_matrix = np.array([[np.cos(theta), -1 * np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0, 0, 1]])
        
        modified_bottom_left_wheel = np.hstack((bottom_left_wheel, np.ones(shape = (bottom_left_wheel.shape[0], 1)))).T
        bottom_left_wheel = (transformation_matrix @ modified_bottom_left_wheel).T 
        bottom_left_wheel = bottom_left_wheel[:, :-1]
        
        modified_bottom_right_wheel = np.hstack((bottom_right_wheel, np.ones(shape = (bottom_right_wheel.shape[0], 1)))).T
        bottom_right_wheel = (transformation_matrix @ modified_bottom_right_wheel).T 
        bottom_right_wheel = bottom_right_wheel[:, :-1]
        
        modified_top_left_wheel = np.hstack((top_left_wheel, np.ones(shape = (top_left_wheel.shape[0], 1)))).T
        top_left_wheel = (transformation_matrix @ modified_top_left_wheel).T 
        top_left_wheel = top_left_wheel[:, :-1]
        
        modified_top_right_wheel = np.hstack((top_right_wheel, np.ones(shape = (top_right_wheel.shape[0], 1)))).T
        top_right_wheel = (transformation_matrix @ modified_top_right_wheel).T 
        top_right_wheel = top_right_wheel[:, :-1]
        
        #Now get an expression for the wheel center of top right and bottom right wheels
        top_right_wheel_center = top_right_wheel_center.reshape((1, -1))
        modified_top_right_wheel_center = np.hstack((top_right_wheel_center, np.ones(shape = (top_right_wheel_center.shape[0], 1)))).T
        top_right_wheel_center = (transformation_matrix @ modified_top_right_wheel_center).T 
        top_right_wheel_center = top_right_wheel_center[:, :-1]
        
        
        bottom_right_wheel_center = bottom_right_wheel_center.reshape((1, -1))
        modified_bottom_right_wheel_center = np.hstack((bottom_right_wheel_center, np.ones(shape = (bottom_right_wheel_center.shape[0], 1)))).T
        bottom_right_wheel_center = (transformation_matrix @ modified_bottom_right_wheel_center).T 
        bottom_right_wheel_center = bottom_right_wheel_center[:, :-1]
        
        #Now modify the top right and bottom right wheels based on steering angles
        steering_transformation_matrix = np.array([[np.cos(phi), -1 * np.sin(phi), top_right_wheel_center[0, 0]], [np.sin(phi), np.cos(phi), top_right_wheel_center[0, 1]], [0, 0, 1]])
        
        modified_top_right_wheel = top_right_wheel - top_right_wheel_center
        modified_top_right_wheel = np.hstack((modified_top_right_wheel, np.ones(shape = (modified_top_right_wheel.shape[0], 1)))).T
        top_right_wheel = (steering_transformation_matrix @ modified_top_right_wheel).T 
        top_right_wheel = top_right_wheel[:, :-1]
        
        
        steering_transformation_matrix = np.array([[np.cos(phi), -1 * np.sin(phi), bottom_right_wheel_center[0, 0]], [np.sin(phi), np.cos(phi), bottom_right_wheel_center[0, 1]], [0, 0, 1]])
        
        modified_bottom_right_wheel = bottom_right_wheel - bottom_right_wheel_center
        modified_bottom_right_wheel = np.hstack((modified_bottom_right_wheel, np.ones(shape = (modified_bottom_right_wheel.shape[0], 1)))).T
        bottom_right_wheel = (steering_transformation_matrix @ modified_bottom_right_wheel).T 
        bottom_right_wheel = bottom_right_wheel[:, :-1]
        
        wheels = [bottom_left_wheel, bottom_right_wheel, top_left_wheel, top_right_wheel]
        return wheels
            
    #Given a configuration, map the body to the workspace
    def generate_rigid_body_from_configuration(self, configuration):
        x, y = configuration[0], configuration[1]
        theta = configuration[2]

        w, h = self.W, self.H
        L = self.L
        
        bottom_left = np.array([L/2 - w/2, -1 * h/2]) 
        bottom_right = np.array([w/2 + L/2, -1 * h/2])
        top_right = np.array([w/2 + L/2, h/2]) 
        top_left = np.array([L/2 - w/2, h/2])
        
        rigid_body = np.vstack((bottom_right, top_right, top_left, bottom_left))
        
        modified_rigid_body = np.hstack((rigid_body, np.ones(shape = (rigid_body.shape[0], 1)))).T
        transformation_matrix = np.array([[np.cos(theta), -1 * np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0, 0, 1]])
        
        new_rigid_body = (transformation_matrix @ modified_rigid_body).T 
        new_rigid_body = new_rigid_body[:, :-1]
        
        return new_rigid_body
    
    #Compute the next configuration given the model governing motion of car
    def compute_next_configuration(self):
        v, phi = self.control_input[0], self.control_input[1]
        theta = self.configuration[2]
        
        first_derivative = np.array([v * np.cos(theta), v * np.sin(theta), v * np.tan(phi)/self.L])
        self.configuration += first_derivative * self.delta_t
        
        #Make sure the theta angle is from -pi to pi
        theta = self.configuration[2]
        theta = theta - (2 * np.pi * (np.floor((theta + np.pi) / (2 * np.pi))))
        self.configuration[2] = theta
    
    #Plot a configuration given a configuration
    def plot_configuration(self, configuration):
        self.rotation_points = np.vstack((self.rotation_points, self.configuration[:2]))
        self.path.set_data(self.rotation_points.T)
        rigid_body = self.generate_rigid_body_from_configuration(configuration)
        wheels = self.generate_wheels_from_configuration(self.configuration)
        
        self.patch.set_xy(rigid_body)
        self.bottom_left_wheel_patch.set_xy(wheels[0])
        self.bottom_right_wheel_patch.set_xy(wheels[1])
        
        self.top_left_wheel_patch.set_xy(wheels[2])
        self.top_right_wheel_patch.set_xy(wheels[3])
        self.f.canvas.draw()
    
    #Update configuration at next time step
    def animation_update_configuration(self, frame):
        old_configuration = np.zeros(shape = self.configuration.shape)
        old_configuration[0] = self.configuration[0]
        old_configuration[1] = self.configuration[1]
        old_configuration[2] = self.configuration[2]
        
        self.compute_next_configuration()
        
        if (self.check_rigid_body_collision(self.generate_rigid_body_from_configuration(self.configuration)) or 
            self.check_wheel_collision(self.generate_wheels_from_configuration(self.configuration))):
            self.configuration[0] = old_configuration[0]
            self.configuration[1] = old_configuration[1]
            self.configuration[2] = old_configuration[2]
                
        self.rotation_points = np.vstack((self.rotation_points, self.configuration[:2]))
        self.path.set_data(self.rotation_points.T)
        
        rigid_body = self.generate_rigid_body_from_configuration(self.configuration)
        wheels = self.generate_wheels_from_configuration(self.configuration)
        
        self.patch.set_xy(rigid_body)
        self.bottom_left_wheel_patch.set_xy(wheels[0])
        self.bottom_right_wheel_patch.set_xy(wheels[1])
        
        self.top_left_wheel_patch.set_xy(wheels[2])
        self.top_right_wheel_patch.set_xy(wheels[3])
        
        print(f"Old Configuration: {old_configuration}, New Configuration: {self.configuration}, Control Input: {self.control_input}, Frame: {frame}")
        return self.patch, self.bottom_left_wheel_patch, self.bottom_right_wheel_patch, self.top_left_wheel_patch, self.top_right_wheel_patch, self.path
        
    # Event handler to change the rotation angle
    def keyboard_event_handler(self, event):
        if event.key == "up":
            self.control_input += np.array([0.1, 0])
        elif event.key == "down":
            self.control_input += np.array([-0.1, 0])
        elif event.key == "right":
            self.control_input += np.array([0, np.pi/12])
        elif event.key == "left":
            self.control_input += np.array([0, -1 * np.pi/12])
        elif event.key == ' ':
            self.control_input += np.array([0,0])
        
        if self.control_input[0] < 0:
            self.control_input[0] = max(self.control_input[0], -0.5)
        elif self.control_input[0] > 0:
            self.control_input[0] = min(self.control_input[0], 0.5)
        
        if self.control_input[1] < 0:
            self.control_input[1] = max(self.control_input[1], -1 * np.pi/4)
        elif self.control_input[1] > 0:
            self.control_input[1] = min(self.control_input[1], 1 * np.pi/4)
        
        old_configuration = np.zeros(shape = self.configuration.shape)
        old_configuration[0] = self.configuration[0]
        old_configuration[1] = self.configuration[1]
        old_configuration[2] = self.configuration[2]
        
        self.compute_next_configuration()
        
        if (self.check_rigid_body_collision(self.generate_rigid_body_from_configuration(self.configuration)) or 
            self.check_wheel_collision(self.generate_wheels_from_configuration(self.configuration))):
            self.configuration[0] = old_configuration[0]
            self.configuration[1] = old_configuration[1]
            self.configuration[2] = old_configuration[2]
        
        self.plot_configuration(self.configuration)
        print(f"Old Configuration: {old_configuration}, New Configuration: {self.configuration}, Control Input: {self.control_input}")
    
    #Function responsible for sampling N random collision-free configurations points
    def sample_configuration_collision_free(self, N):
        #Sample a configuration, uniform at random 
        #Configuration is of the form [x, y, theta] where (x, y) is the location of the focal center of the body and theta is the orientation
        overall_configurations = []
        sampled_configurations = []
        
        P = 0
        iterations = 0
        while P < N:
            #Sample Configuration
            configuration = np.array([np.random.uniform(0, 2), np.random.uniform(0, 2), np.random.uniform(-1 * np.pi, np.pi)])
            overall_configurations.append(configuration)
            
            #Generate Rigid Body and Wheels from Configuration            
            rigid_body = self.generate_rigid_body_from_configuration(configuration)
            wheels = self.generate_wheels_from_configuration(configuration)
            
            #If the rigid body does not collide with anything in the workspace, we have sampled a valid configuration in free C-space
            if ((not self.check_rigid_body_collision(rigid_body)) and (not self.check_wheel_collision(wheels))):
                P = P + 1
                sampled_configurations.append(configuration)
            
            iterations = iterations + 1
        
        return np.array(sampled_configurations)
    
    #Function Responsible for Sampling B random controls
    def sample_controls(self, B):
        sampled_controls = []
        for _ in range(B):
            u = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)])
            sampled_controls.append(u)
        
        return np.array(sampled_controls)    
    
#Class Representing a Rapidly Exploring Random Tree
class RRT:
    def __init__(self, start, goal, rigid_body: Car, b, num_integrations = 40):
        #Set up start and goal nodes
        self.start = start.flatten()
        self.goal = goal.flatten()
        
        #Set up other hyperparameters for RRT Generation
        self.num_integrations = num_integrations
        self.b = b
        
        #Set up vertices and edges and integration steps
        self.vertices = np.ones(shape = (1, start.shape[0]))
        self.vertices[0] = self.start
        self.edges = np.zeros(shape = (len(self.vertices), len(self.vertices)))
        self.control_information = dict()
        self.integration_information = dict()
        
        #Set up rigid body
        self.rigid_body = rigid_body
        
        #Empty Dictionary to store path in tree
        self.predecessor = {}
        
        #Variable telling if we have sampled goal vertex
        self.sampled_goal = False
        self.goal_index = None
        
        #Set up Animation Stuff
        #Animation Object
        self.animation = None
        self.animation_configuration = self.start
        
        #Centroid Points
        self.centroid_points = np.empty(shape = (0, 2))
        
        #Set up centroid of rigid body
        self.body_centroid = self.rigid_body.ax.plot(self.start[0], self.start[1], marker='o', markersize=3, color="green")
        self.frame_number = 0
        
    #Add vertex where vertex is a C space point
    def add_vertex(self, vertex):
        #Find closest vertex on RRT
        vertex = vertex.flatten()
        qrand = vertex
        
        closest_vertex_index = np.argmin(np.apply_along_axis(func1d = self.D, axis = 1, arr = self.vertices - vertex.reshape((1, vertex.shape[0]))))
        qclose = self.vertices[closest_vertex_index].flatten()
        
        #Sample Controls
        sampled_controls = self.rigid_body.sample_controls(self.b)
        
        #For each of the sampled controls, we will integrate forward in time the dynamics from the closest node qclose for the same duration T
        #We will get b new states that are reachable from qclose
        new_configurations = []
        integrations_per_control = []
        
        #Indicate if we have found valid neighbors
        valid_neighbors_found = False
        
        for control in sampled_controls:
            u = control.flatten()
            
            q_prev = qclose
            q_new = qclose
            self.rigid_body.set_state(qclose[0], qclose[1], qclose[2])
            self.rigid_body.set_control_input(u[0], u[1])
            
            is_valid_control = False
            
            integrations = 0
            for _ in range(self.num_integrations):
                self.rigid_body.compute_next_configuration()
            
                q_new = self.rigid_body.configuration.flatten()
                rigid_body_workspace = self.rigid_body.generate_rigid_body_from_configuration(q_new)
                wheels_workspace = self.rigid_body.generate_wheels_from_configuration(q_new)
                
                if (self.rigid_body.check_rigid_body_collision(rigid_body_workspace) or self.rigid_body.check_wheel_collision(wheels_workspace)):
                    q_new = q_prev
                    break
                else:
                    is_valid_control = True
                    integrations += 1
                        
            if (not is_valid_control):
                new_configurations.append(np.array([np.inf, np.inf, np.inf]))
                integrations_per_control.append(0)
            else:
                new_configurations.append(q_new)
                integrations_per_control.append(integrations)
                valid_neighbors_found = True
        
        index = None
        
        #Identify which of the b new states is closest to qrand 
        if valid_neighbors_found:
            new_configurations = np.array(new_configurations)
            closest_to_q_rand_index = np.argmin(np.apply_along_axis(func1d = self.D, axis = 1, arr = new_configurations - qrand.reshape((1, qrand.shape[0]))))
            q_new_closest = new_configurations[closest_to_q_rand_index].flatten()
            
            ### Add q_new_closest to RRT and Edge (q_close, q_new_closest)
            self.vertices = np.append(self.vertices, q_new_closest.reshape((1, q_new_closest.shape[0])), axis = 0)
            self.edges = np.vstack((self.edges, np.zeros(shape = (1, self.edges.shape[1]))))
            self.edges = np.hstack((self.edges, np.zeros(shape = (self.edges.shape[0], 1))))
            
            self.edges[closest_vertex_index, -1] = 1
            self.edges[-1, closest_vertex_index] = 1  
            
            self.predecessor[len(self.vertices) - 1] = closest_vertex_index
            
            self.control_information[(closest_vertex_index, len(self.vertices) - 1)] = sampled_controls[closest_to_q_rand_index]
            self.control_information[(len(self.vertices) - 1, closest_vertex_index)] = sampled_controls[closest_to_q_rand_index]
            
            self.integration_information[(closest_vertex_index, len(self.vertices) - 1)] = integrations_per_control[closest_to_q_rand_index]
            self.integration_information[(len(self.vertices) - 1, closest_vertex_index)] = integrations_per_control[closest_to_q_rand_index]
            
            #If we have reached goal, set some indicators
            if q_new_closest[0] == self.goal[0] and q_new_closest[1] == self.goal[1] and q_new_closest[2] == self.goal[2]:
                self.sampled_goal = True
                self.goal_index = len(self.vertices) - 1
            
            index = len(self.vertices) - 1
        
        return valid_neighbors_found, index
    
    #define our Distance Function
    def D(self, point):
        #Flatten
        point = point.flatten()
        
        #Calculate Euclidean Distance Transitionally
        dt = np.linalg.norm(point[:-1])
        
        #Calculate Rotational Distance
        angle = point[-1]
        dr = min(abs(angle), 2 * np.pi - abs(angle))
        
        return 0.7 * dt + 0.3 * dr
    
    #Generate Path from start to goal
    #The path are indices so its easier
    def generate_path(self, goal_index):
        self.goal_index = goal_index
        path = [self.goal_index]
        current_configuration = self.goal.flatten()
        current_configuration_index = self.goal_index
        
        controls = []
        integration_steps = []
        
        while not self.equal(current_configuration, self.start):
            next_configuration_index = self.predecessor.get(current_configuration_index, None)
            
            u = self.control_information[(next_configuration_index, current_configuration_index)].flatten()
            controls.append(u)
            
            steps = self.integration_information[(next_configuration_index, current_configuration_index)]
            integration_steps.append(steps)
            
            if next_configuration_index == None: 
                return np.empty(shape = (0,0)), np.empty(shape = (0,0)), np.empty(shape = (0,0))
            
            path.append(next_configuration_index)
            current_configuration = self.vertices[next_configuration_index]
            current_configuration_index = next_configuration_index
        
        path = np.flip(np.array(path))
        self.rrt_path = path     
        
        controls = np.flip(np.array(controls), axis = 0)
        self.control_sequence = controls
        
        integration_steps = np.flip(np.array(integration_steps))
        self.integration_steps = integration_steps
        
        return path, controls, integration_steps
        
    #Return true if two configurations are equal
    def equal(self, configurationA, configurationB):
        configurationA, configurationB = configurationA.flatten(), configurationB.flatten()
        return configurationA[0] == configurationB[0] and configurationA[1] == configurationB[1] and configurationA[2] == configurationB[2]
    
    #Function for init animation
    def init_animation_configuration(self):
        #Set animation configuration to start node
        self.animation_configuration = self.start
        configuration = self.animation_configuration
        
        #Set control to 1st control in control sequence
        u = self.control_sequence[0].flatten()
        self.rigid_body.set_state(configuration[0], configuration[1], configuration[2])
        self.rigid_body.set_control_input(u[0], u[1])
        
        #Get Car and Wheels
        rigid_body_workspace = self.rigid_body.generate_rigid_body_from_configuration(configuration)
        wheels = self.rigid_body.generate_wheels_from_configuration(configuration)
        
        #Update Rigid Body Patch
        self.rigid_body.patch.set_xy(rigid_body_workspace)
        
        #Add wheel patches to rigid body figure
        self.rigid_body.bottom_left_wheel_patch.set_xy(wheels[0])
        self.rigid_body.bottom_right_wheel_patch.set_xy(wheels[1])
        self.rigid_body.top_left_wheel_patch.set_xy(wheels[2])
        self.rigid_body.top_right_wheel_patch.set_xy(wheels[3])
        
        #Get Centroid Points and add to rigid body figure
        self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
        self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
        self.rigid_body.ax.add_line(self.path)
        self.body_centroid[0].set_data([configuration[0], configuration[1]])
        
        return self.rigid_body.patch, self.rigid_body.bottom_left_wheel_patch, self.rigid_body.bottom_right_wheel_patch, self.rigid_body.top_left_wheel_patch, self.rigid_body.top_right_wheel_patch, self.path, self.body_centroid[0]
    
    #Function responsible for plotting a configuration
    def update_animation_configuration(self, frame):        
        #Get Configuration
        configuration = self.animation_configuration.flatten()
        
        #Compute subsequence sum of integration step array
        integration_step_index = None
        subsequence_total_steps_array = [-1]
        for index in range(len(self.integration_steps)):
            subsequence_total_steps = np.sum(self.integration_steps[0: index + 1])
            subsequence_total_steps_array.append(subsequence_total_steps)
        
        #Get the integration step index
        for index in range(len(self.integration_steps)):
            steps = subsequence_total_steps_array[index + 1]
            prev_steps = subsequence_total_steps_array[index]
            
            if prev_steps < frame and frame <= steps:
                integration_step_index = index
                break
        
        u = self.control_sequence[integration_step_index].flatten()
        # print(f"Frame: {frame}, Configuration: {configuration}, Integration Step Index: {integration_step_index}, Control: {u}")
        
        self.rigid_body.set_state(configuration[0], configuration[1], configuration[2])
        self.rigid_body.set_control_input(u[0], u[1])
        
        #Get Car and Wheels
        rigid_body_workspace = self.rigid_body.generate_rigid_body_from_configuration(configuration)
        wheels = self.rigid_body.generate_wheels_from_configuration(configuration)
        
        #Update Rigid Body Patch
        self.rigid_body.patch.set_xy(rigid_body_workspace)
        
        #Add wheel patches to rigid body figure
        self.rigid_body.bottom_left_wheel_patch.set_xy(wheels[0])
        self.rigid_body.bottom_right_wheel_patch.set_xy(wheels[1])
        self.rigid_body.top_left_wheel_patch.set_xy(wheels[2])
        self.rigid_body.top_right_wheel_patch.set_xy(wheels[3])
        
        #Get Centroid Points and add to rigid body figure
        self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
        self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
        self.rigid_body.ax.add_line(self.path)
        
        self.body_centroid[0].set_data([configuration[0], configuration[1]])
        self.frame_number = frame
        
        #Calculate Next Configuration
        original_configuration = self.rigid_body.configuration
        self.rigid_body.compute_next_configuration()
        self.animation_configuration = self.rigid_body.configuration
        # print(f"[2]Frame: {frame}, Original Configuration: {original_configuration}, Next Computed Configuration: {self.animation_configuration}")
        
        return self.rigid_body.patch, self.rigid_body.bottom_left_wheel_patch, self.rigid_body.bottom_right_wheel_patch, self.rigid_body.top_left_wheel_patch, self.rigid_body.top_right_wheel_patch, self.path, self.body_centroid[0]