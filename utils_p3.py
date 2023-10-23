### In this file, I defined utility functions for collision checking, Rigid Body for Problem 3
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt

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
    def __init__(self, f, ax, file):
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
        
        #Set up configuration and control input for car
        self.W = 0.2
        self.H = 0.1
        self.L = 0.2
        self.configuration = self.initialize_configuration()
        self.control_input = np.zeros(shape = (2,))
        
        #Set up delta_t information
        self.delta_t = 0.001
        
        #Register Keyboard Handler
        self.f.canvas.mpl_connect('key_press_event', self.keyboard_event_handler)
        
        #Plot Car based on initial configuration
        rigid_body = self.generate_rigid_body_from_configuration(self.configuration)
        self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')
        self.ax.add_patch(self.patch)
    
    #Set State
    def set_state(self, x, y, theta):
        self.configuration = np.array([x, y, theta])
    
    #Set Control Input
    def set_control_input(self, v, phi):
        self.control_input = np.array([v, np.rad2deg(phi)])
    
    #Returns True if rigid body is on boundary of discrete grid environment, Else returns False
    def is_rigid_body_on_boundary(self, rigid_body):
        for vertex in rigid_body:
            if vertex[0] <= 0 or vertex[0] >= 2 or vertex[1] <= 0 or vertex[1] >= 2:
                return True
        return False

    #Check to see if the Selected Configuration collides with other C-obstacles
    #This entails to checking to see if the Rigid Body collides with other polygons in our list or if the rigid boundary is on the boundary
    #Returns True if there is a collision. Else, returns False
    def check_configuration_collision(self, rigid_body):
        for polygon in self.polygonal_obstacles:
            if (check_polygon_collision(polygon, np.vstack((rigid_body, rigid_body[0])))):
                return True
        
        return self.is_rigid_body_on_boundary(rigid_body)
    
    #Initialize the configuration of car
    def initialize_configuration(self):
        configuration = np.array([np.random.uniform(0, 2, size = None), np.random.uniform(0, 2, size = None), np.random.uniform(0, 360, size = None)]) #randomly choose a configuration        
        free = False #determines whether the configuration is collision free
        
        while not free:
            rigid_body = self.generate_rigid_body_from_configuration(configuration)
            if not self.check_configuration_collision(rigid_body):
                free = True
                break
            
            configuration = np.array([np.random.uniform(0, 2, size = None), np.random.uniform(0, 2, size = None), np.random.uniform(0, 360, size = None)]) #randomly choose a configuration
        
        return configuration
    
    #Given a configuration, map the body to the workspace
    def generate_rigid_body_from_configuration(self, configuration):
        x, y = configuration[0], configuration[1]
        theta = np.deg2rad(configuration[2])

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
        
        first_derivative = np.array([v * np.cos(np.deg2rad(theta)), v * np.sin(np.deg2rad(theta)), v * np.tan(np.deg2rad(phi))/self.L])
        self.configuration += first_derivative * self.delta_t
        
        #Reformat theta
        self.configuration[2] = np.rad2deg(np.deg2rad(theta)  + (first_derivative * self.delta_t)[2])
        self.configuration[2] = self.configuration[2] + 360 if self.configuration[2] < 0 else self.configuration[2]
        self.configuration[2] = self.configuration[2] % 360
    
    #Plot a configuration given a configuration
    def plot_configuration(self, configuration):
        rigid_body = self.generate_rigid_body_from_configuration(configuration)
        self.patch.set_xy(rigid_body)
        self.f.canvas.draw()
    
    #Update configuration at next time step
    def update_configuration(self, frame):
        old_configuration = np.zeros(shape = self.configuration.shape)
        old_configuration[0] = self.configuration[0]
        old_configuration[1] = self.configuration[1]
        old_configuration[2] = self.configuration[2]
        
        self.compute_next_configuration()
        
        if (self.check_configuration_collision(self.generate_rigid_body_from_configuration(self.configuration))):
            self.configuration[0] = old_configuration[0]
            self.configuration[1] = old_configuration[1]
            self.configuration[2] = old_configuration[2]
        
        rigid_body = self.generate_rigid_body_from_configuration(self.configuration)
        self.patch.set_xy(rigid_body)
        print(f"Old Configuration: {old_configuration}, New Configuration: {self.configuration}, Control Input: {self.control_input}")
        return self.patch,
        
    # Event handler to change the rotation angle
    def keyboard_event_handler(self, event):
        if event.key == "up":
            self.control_input += np.array([0.01, 0])
        elif event.key == "down":
            self.control_input += np.array([-0.01, 0])
        elif event.key == "right":
            self.control_input += np.array([0, 15])
        elif event.key == "left":
            self.control_input += np.array([0, -15])
        
        if self.control_input[0] < 0:
            self.control_input[0] = max(self.control_input[0], -0.5)
        elif self.control_input[0] > 0:
            self.control_input[0] = min(self.control_input[0], 0.5)
        
        if self.control_input[1] < 0:
            self.control_input[1] = np.rad2deg(max(np.deg2rad(self.control_input[1]), -1 * np.pi/4))
        elif self.control_input[1] > 0:
            self.control_input[1] = np.rad2deg(min(np.deg2rad(self.control_input[1]), np.pi/4))
        
        old_configuration = np.zeros(shape = self.configuration.shape)
        old_configuration[0] = self.configuration[0]
        old_configuration[1] = self.configuration[1]
        old_configuration[2] = self.configuration[2]
        
        self.compute_next_configuration()
        
        if (self.check_configuration_collision(self.generate_rigid_body_from_configuration(self.configuration))):
            self.configuration[0] = old_configuration[0]
            self.configuration[1] = old_configuration[1]
            self.configuration[2] = old_configuration[2]
        
        self.plot_configuration(self.configuration)
        print(f"Old Configuration: {old_configuration}, New Configuration: {self.configuration}, Control Input: {self.control_input}")
    
    
