### Utility Functions
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

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
    
    
# Python Class For 2D Rigid Body
class RigidBody:
    #Responsible for loading up environment with grid-discretization and polygonal obstacles
    def __init__(self, f, ax, file):
        #Store figure and axes as instance variables
        self.f = f
        self.ax = ax
        
        #set axis limits for x and y axes
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        
        #set title for axis
        ax.set_title('2D Rigid Body Simulation', fontsize = 12)
        
        #Load Polygon Obstacle Data
        self.polygonal_obstacles = np.load(file, allow_pickle= True)

        #Plot polygons from the npy file on the grid
        for index in range(len(self.polygonal_obstacles)):
            self.ax.fill([vertex[0] for vertex in self.polygonal_obstacles[index]], [vertex[1] for vertex in self.polygonal_obstacles[index]], alpha=.25, fc='white', ec='black')

    #Returns True if rigid body is on boundary of discrete grid environment, Else returns False
    def is_rigid_body_on_boundary(self, rigid_body):
        for vertex in rigid_body:
            if vertex[0] <= 0 or vertex[0] >= 2 or vertex[1] <= 0 or vertex[1] >= 2:
                return True
        return False
        
    #Function responsible for sampling N random collision-free configurations points
    def sample_configuration_collision_free(self, N):
        #Sample a configuration, uniform at random 
        #Configuration is of the form [x, y, theta] where (x, y) is the location of the geometric center of the body and theta is the orientation
        sampled_configurations = []
        
        P = 0
        while P < N:
            #Sample Configuration
            configuration = np.array([np.random.uniform(0, 2), np.random.uniform(0, 2), np.random.uniform(0, 360)])
            
            #Construct Rigid Body in the workspace
            w = 0.2
            h = 0.1
            top_left = np.array([-1 * w/2, h/2])
            top_right = np.array([w/2, h/2])
            bottom_left = np.array([-1 * w/2, -1 * h/2])
            bottom_right = np.array([w/2, -1 * h/2])
            
            rigid_body = np.vstack((bottom_right, top_right, top_left, bottom_left))
            rigid_body = np.hstack((rigid_body, np.ones(shape = (rigid_body.shape[0], 1)))).T
            
            #Construct Rotation Matrix from configuration
            angle = np.deg2rad(configuration[2])
            transformation_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), configuration[0]], [np.sin(angle), np.cos(angle), configuration[1]], [0, 0, 1]])
            
            #Calculate final workspace coordinates
            rigid_body = ((transformation_matrix @ rigid_body).T)[:, :-1]
            
            if (not self.check_configuration_collision(rigid_body)):
                P = P + 1
                sampled_configurations.append(configuration)
        
        return sampled_configurations
    
    #Function responsible for plotting a configuration
    def plot_configuration(self, configuration):
        #Construct Rigid Body in the workspace
        w = 0.2
        h = 0.1
        top_left = np.array([-1 * w/2, h/2])
        top_right = np.array([w/2, h/2])
        bottom_left = np.array([-1 * w/2, -1 * h/2])
        bottom_right = np.array([w/2, -1 * h/2])
        
        rigid_body = np.vstack((bottom_right, top_right, top_left, bottom_left))
        rigid_body = np.hstack((rigid_body, np.ones(shape = (rigid_body.shape[0], 1)))).T
        
        #Construct Rotation Matrix from configuration
        angle = np.deg2rad(configuration[2])
        transformation_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), configuration[0]], [np.sin(angle), np.cos(angle), configuration[1]], [0, 0, 1]])
        
        #Calculate final workspace coordinates
        rigid_body = ((transformation_matrix @ rigid_body).T)[:, :-1]
        
        rectangle_patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')
        self.ax.add_patch(rectangle_patch)

        # Plot centroid of rectangle
        body_centroid = self.ax.plot(configuration[0],configuration[1], marker='o', markersize=3, color="green")
    
    #Check to see if the Selected Configuration collides with other C-obstacles
    #This entails to checking to see if the Rigid Body collides with other polygons in our list or if the rigid boundary is on the boundary
    #Returns True if there is a collision. Else, returns False
    def check_configuration_collision(self, rigid_body):
        for polygon in self.polygonal_obstacles:
            if (check_polygon_collision(polygon, np.vstack((rigid_body, rigid_body[0])))):
                return True
        
        return self.is_rigid_body_on_boundary(rigid_body)