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
        
        #Load Polygon Obstacle Data if the file has a value
        if file != None:
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
        
        return np.array(sampled_configurations)
    
    #Function responsible for plotting a configuration
    def plot_configuration(self, configuration, color = 'r'):
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
        
        rectangle_patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = color, edgecolor = color)
        self.ax.add_patch(rectangle_patch)

        # Plot centroid of rectangle
        body_centroid = self.ax.plot(configuration[0],configuration[1], marker='o', markersize=3, color="green")
    
    #Generate a Rigid Body from the Configuration Information
    def generate_rigid_body_from_configuration(self, configuration):
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
        return rigid_body
    
    #Check to see if the Selected Configuration collides with other C-obstacles
    #This entails to checking to see if the Rigid Body collides with other polygons in our list or if the rigid boundary is on the boundary
    #Returns True if there is a collision. Else, returns False
    def check_configuration_collision(self, rigid_body):
        for polygon in self.polygonal_obstacles:
            if (check_polygon_collision(polygon, np.vstack((rigid_body, rigid_body[0])))):
                return True
        
        return self.is_rigid_body_on_boundary(rigid_body)
  
#Class Representing a Rapidly Exploring Random Tree
class RRT:
    def __init__(self, start, goal, rigid_body = None):
        self.start = start.flatten()
        self.goal = goal.flatten()
        
        self.vertices = np.ones(shape = (1, start.shape[0]))
        self.vertices[0] = self.start
        
        self.edges = np.zeros(shape = (len(self.vertices), len(self.vertices)))
        self.rigid_body = rigid_body
        
    #Add vertex where vertex is a C space point
    def add_vertex(self, vertex):
        vertex = vertex.flatten()
        
        closest_vertex_index = np.argmin(np.apply_along_axis(func1d = self.D, axis = 1, arr = self.vertices - vertex.reshape((1, vertex.shape[0]))))
        closest_vertex = self.vertices[closest_vertex_index].flatten()
                
        #Check path from closest_vertex to vertex
        timesteps = 5
        new_vertex = None
        valid_path_found = False
        for timestep in range(1, timesteps + 1):
            configuration = closest_vertex + ((vertex - closest_vertex) * timestep / timesteps)
            if self.rigid_body.check_configuration_collision(self.rigid_body.generate_rigid_body_from_configuration(configuration)):
                break
            else:
                new_vertex = configuration
                valid_path_found = True
        
        if valid_path_found:
            self.vertices = np.append(self.vertices, new_vertex.reshape((1, new_vertex.shape[0])), axis = 0)
            self.edges = np.vstack((self.edges, np.zeros(shape = (1, self.edges.shape[1]))))
            self.edges = np.hstack((self.edges, np.zeros(shape = (self.edges.shape[0], 1))))

            self.edges[closest_vertex_index, -1] = 1
            self.edges[-1, closest_vertex_index] = 1    
    
    def D(self, point):
        point = point.flatten()
        return 0.7 * np.linalg.norm(point[:-1]) + 0.3 * np.deg2rad(np.abs(point[-1]))


#Class Representing Probabilistic Road Map
class PRM:
    def __init__(self, N, rigid_body = None):        
        self.rigid_body = rigid_body
        self.vertices = rigid_body.sample_configuration_collision_free(N)
        self.edges = np.zeros(shape = (N, N))
        
    def compute_edges(self):
        k = 3
        for index in range(len(self.vertices)):
            target_configuration = self.vertices[index].flatten()
            configurations = self.vertices
            neighbor_distances = np.apply_along_axis(func1d = self.D, axis = 1, arr = configurations - target_configuration)
            neighbor_indices = np.delete(np.argsort(neighbor_distances), index) [:k]
            
            for neighbor_index in neighbor_indices:
                if self.is_edge_valid(target_configuration, self.vertices[neighbor_index]):
                    self.edges[index, neighbor_index] = 1
                    self.edges[neighbor_index, index] = 1
                    break
        
    
    #Tell if Edge is valid based on collision checking in workspace
    def is_edge_valid(self, vertexA, vertexB):
        vertexA = vertexA.flatten()
        vertexB = vertexB.flatten()
        
        #Check path from vertexA to vertexB
        timesteps = 5
        is_valid_path = True
        
        for timestep in range(1, timesteps + 1):
            configuration = vertexA + ((vertexB - vertexA) * timestep / timesteps)
            if self.rigid_body.check_configuration_collision(self.rigid_body.generate_rigid_body_from_configuration(configuration)):
                is_valid_path = False
                break
        
        return is_valid_path

    #Shortest path from start -> goal
    def answer_query(self, start, goal):
        k = 3
        target_configurations = [start, goal]
        for target_configuration in target_configurations:
            configurations = self.vertices
            neighbor_distances = np.apply_along_axis(func1d = self.D, axis = 1, arr = configurations - target_configuration)
            neighbor_indices = np.argsort(neighbor_distances) [:k]
            
            for neighbor_index in neighbor_indices:
                if self.is_edge_valid(target_configuration, self.vertices[neighbor_index]):
                    self.vertices = np.vstack((self.vertices, target_configuration.reshape((1, -1))))
                    self.edges[-1, neighbor_index] = 1
                    self.edges[neighbor_index, -1] = 1
                    break
        
        #RUN SEARCH ALGORITHM HERE
            
    def D(self, point):
        point = point.flatten()
        return 0.7 * np.linalg.norm(point[:-1]) + 0.3 * np.deg2rad(np.abs(point[-1]))