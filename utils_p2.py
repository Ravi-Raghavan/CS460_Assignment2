### In this file, I defined utility functions for collision checking, Rigid Body for Problem 2, RRT and PRM for Problem 2
import numpy as np 
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from queue import PriorityQueue


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
    def __init__(self, f, ax, file, starting_configuration = None, goal_configuration = None, timesteps = None):
        #Store figure and axes as instance variables
        self.f = f
        self.ax = ax
        
        #set axis limits for x and y axes
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        
        #set title for axis
        ax.set_title('2D Rigid Body', fontsize = 12)
        
        #Load Polygon Obstacle Data if the file has a value
        self.polygonal_obstacles = np.empty(shape = (0,0))
        if file != None:
            self.polygonal_obstacles = np.load(file, allow_pickle= True)

            #Plot polygons from the npy file on the grid
            for index in range(len(self.polygonal_obstacles)):
                self.ax.fill([vertex[0] for vertex in self.polygonal_obstacles[index]], [vertex[1] for vertex in self.polygonal_obstacles[index]], alpha=.25, fc='white', ec='black')
        
        #Set a baseline animation configuration 
        self.start_configuration = starting_configuration
        self.goal_configuration = goal_configuration
        self.timesteps = timesteps
        self.animation_configuration = self.start_configuration
        
        if isinstance(self.start_configuration, np.ndarray):
            rigid_body = self.generate_rigid_body_from_configuration(self.start_configuration)
            self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')
            self.ax.add_patch(self.patch)
            self.body_centroid = self.ax.plot(self.start_configuration[0], self.start_configuration[1], marker='o', markersize=3, color="magenta")
            self.centroid_points = np.empty(shape = (0, 2))
            self.centroid_points = np.vstack((self.centroid_points, self.start_configuration[:2]))
            
            self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
            self.ax.add_line(self.path)

    #Returns True if rigid body is on boundary of discrete grid environment, Else returns False
    def is_rigid_body_on_boundary(self, rigid_body):
        for vertex in rigid_body:
            if vertex[0] <= 0 or vertex[0] >= 2 or vertex[1] <= 0 or vertex[1] >= 2:
                return True
        return False
    
    #Check to see if the rigid body collides with other polygons in workspace or collides with the boundary
    #This entails to checking to see if the Rigid Body collides with other polygons in our list or if the rigid boundary is on the boundary
    #Returns True if there is a collision. Else, returns False
    def check_rigid_body_collision(self, rigid_body):
        for polygon in self.polygonal_obstacles:
            if (check_polygon_collision(polygon, np.vstack((rigid_body, rigid_body[0])))):
                return True
        
        return self.is_rigid_body_on_boundary(rigid_body)
        
    #Function responsible for sampling N random collision-free configurations points
    def sample_configuration_collision_free(self, N):
        #Sample a configuration, uniform at random 
        #Configuration is of the form [x, y, theta] where (x, y) is the location of the geometric center of the body and theta is the orientation
        sampled_configurations = []
        P = 0
        while P < N:
            #Sample Configuration
            configuration = np.array([np.random.uniform(0, 2), np.random.uniform(0, 2), np.random.uniform(-1 * np.pi, np.pi)])
            
            #Generate Rigid Body from Configuration            
            rigid_body_workspace = self.generate_rigid_body_from_configuration(configuration)
            
            #If the rigid body does not collide with anything in the workspace, we have sampled a valid configuration in free C-space
            if (not self.check_rigid_body_collision(rigid_body_workspace)):
                P = P + 1
                sampled_configurations.append(configuration)
        
        return np.array(sampled_configurations)
    
    #Function reponsible for initializing animation
    def init_animation_configuration(self):
        self.animation_configuration = self.start_configuration
        configuration = self.start_configuration
        
        #Generate Rigid Body from Configuration and Plot in Workspace        
        rigid_body_workspace = self.generate_rigid_body_from_configuration(configuration)  
        
        if hasattr(self, 'patch'):      
            self.patch.set_xy(rigid_body_workspace)
        else:
            self.patch = matplotlib.patches.Polygon(rigid_body_workspace, closed=True, facecolor = 'none', edgecolor='r')

        # Plot Centroid of rectangle
        if hasattr(self, 'body_centroid'):      
            self.body_centroid[0].set_data([configuration[0], configuration[1]])
            self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
            self.path.set_data(self.centroid_points.T)
        else:
            self.body_centroid = self.ax.plot(configuration[0], configuration[1], marker='o', markersize=3, color="green")
            self.centroid_points = np.empty(shape = (0, 2))
            self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
            self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
            
        print(f"Centroid Points: {self.centroid_points}")
        
        return self.patch, self.path, self.body_centroid[0]
    
    #Function responsible for plotting a configuration
    def update_animation_configuration(self, frame):
        self.start_configuration = self.start_configuration.flatten()
        self.goal_configuration = self.goal_configuration.flatten()        
        dConfiguration = (self.goal_configuration - self.start_configuration) / self.timesteps
        
        DeltaTheta = self.goal_configuration[2] - self.start_configuration[2]
        if DeltaTheta > np.pi:
            DeltaTheta = DeltaTheta - (2 * np.pi)
        elif DeltaTheta < -1 * np.pi:
            DeltaTheta = (2 * np.pi) - DeltaTheta
            
        dTheta = DeltaTheta / self.timesteps
        dConfiguration[2] = dTheta
        
        self.animation_configuration += dConfiguration
        
        #Make sure config angle is within range
        self.animation_configuration[2] =  self.animation_configuration[2] - (2 * np.pi * (np.floor(( self.animation_configuration[2] + np.pi) / (2 * np.pi))))
        configuration = self.animation_configuration
        
        #Generate Rigid Body from Configuration and Plot in Workspace        
        rigid_body = self.generate_rigid_body_from_configuration(configuration)  
        
        if hasattr(self, 'patch'):      
            self.patch.set_xy(rigid_body)
        else:
            self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')

        # Plot Centroid of rectangle
        if hasattr(self, 'body_centroid'):      
            self.body_centroid[0].set_data([configuration[0], configuration[1]])
            self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
            self.path.set_data(self.centroid_points.T)
        else:
            self.body_centroid = self.ax.plot(configuration[0], configuration[1], marker='o', markersize=3, color="green")
            self.centroid_points = np.empty(shape = (0, 2))
            self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
            self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
            
        print(f"Centroid Points: {self.centroid_points}")
        return self.patch, self.path, self.body_centroid[0]
    
    def plot_configuration(self, configuration, color = 'r'):
        #Generate Rigid Body from Configuration and Plot in Workspace        
        rigid_body = self.generate_rigid_body_from_configuration(configuration)
        rectangle_patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = color, edgecolor = color)
        self.ax.add_patch(rectangle_patch)

        # Plot Centroid of rectangle
        body_centroid = self.ax.plot(configuration[0], configuration[1], marker='o', markersize=3, color="magenta")
                
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
        angle = configuration[2]
        transformation_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), configuration[0]], [np.sin(angle), np.cos(angle), configuration[1]], [0, 0, 1]])
        
        #Calculate final workspace coordinates
        rigid_body = ((transformation_matrix @ rigid_body).T)[:, :-1]
        return rigid_body
  
#Class Representing a Rapidly Exploring Random Tree
class RRT:
    def __init__(self, start, goal, rigid_body: RigidBody):
        #Set up start and goal nodes
        self.start = start.flatten()
        self.goal = goal.flatten()
        
        #Set up vertices and edges
        self.vertices = np.ones(shape = (1, start.shape[0]))
        self.vertices[0] = self.start
        self.edges = np.zeros(shape = (len(self.vertices), len(self.vertices)))
        
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
        
        #Centroid Points
        self.centroid_points = np.empty(shape = (0, 2))
        
        #Set up centroid of rigid body
        self.body_centroid = self.rigid_body.ax.plot(self.start[0], self.start[1], marker='o', markersize=3, color="green")
        
        self.frame_number = 0
        
    #Add vertex where vertex is a C space point
    def add_vertex(self, vertex):
        #Find closest vertex on RRT
        vertex = vertex.flatten()
        closest_vertex_index = np.argmin(np.apply_along_axis(func1d = self.D, axis = 1, arr = self.vertices - vertex.reshape((1, vertex.shape[0]))))
        closest_vertex = self.vertices[closest_vertex_index].flatten()
        
        #Check path from closest_vertex to vertex
        timesteps = 100
        valid_path_found = False
        current_configuration = closest_vertex
        
        for timestep in range(1, timesteps + 1):
            dConfiguration = ((vertex - closest_vertex) / timesteps)
            
            DeltaTheta = vertex[2] - closest_vertex[2]
            if DeltaTheta > np.pi:
                DeltaTheta = DeltaTheta - (2 * np.pi)
            elif DeltaTheta < -1 * np.pi:
                DeltaTheta = (2 * np.pi) - DeltaTheta
                
            dTheta = DeltaTheta / timesteps
            dConfiguration[2] = dTheta
        
            next_configuration = current_configuration + dConfiguration
            
            #Make angle within range
            next_configuration[2] = next_configuration[2] - (2 * np.pi * (np.floor((next_configuration[2] + np.pi) / (2 * np.pi))))
            
            if self.rigid_body.check_rigid_body_collision(self.rigid_body.generate_rigid_body_from_configuration(next_configuration)):
                break
            else:
                valid_path_found = True
                current_configuration = next_configuration
        
        #If we found valid path, add to graph
        if valid_path_found:            
            self.vertices = np.append(self.vertices, current_configuration.reshape((1, current_configuration.shape[0])), axis = 0)
            self.edges = np.vstack((self.edges, np.zeros(shape = (1, self.edges.shape[1]))))
            self.edges = np.hstack((self.edges, np.zeros(shape = (self.edges.shape[0], 1))))
            
            self.edges[closest_vertex_index, -1] = 1
            self.edges[-1, closest_vertex_index] = 1  
            
            self.predecessor[len(self.vertices) - 1] = closest_vertex_index
            
            if current_configuration[0] == self.goal[0] and current_configuration[1] == self.goal[1] and current_configuration[2] == self.goal[2]:
                self.sampled_goal = True
                self.goal_index = len(self.vertices) - 1
                print("We have sampled the goal vertex")
        
        return valid_path_found
    
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
    def generate_path(self):
        if not self.sampled_goal:
            print("We have not sampled the goal vertex. Instead, we will plot the path to the point closest to the goal node")
            self.goal = self.goal.flatten()
            closest_vertex_index = np.argmin(np.apply_along_axis(func1d = self.D, axis = 1, arr = self.vertices - self.goal.reshape((1, self.goal.shape[0]))))
            
            #Update goal and goal_index
            self.goal_index = closest_vertex_index
            self.goal = self.vertices[closest_vertex_index].flatten()
            
            #New Goal
            print(f"The closest Node we could find to the goal node was {self.goal}")
        
            
        path = [self.goal_index]
        current_configuration = self.goal.flatten()
        current_configuration_index = self.goal_index
        
        while not self.equal(current_configuration, self.start):
            next_configuration_index = self.predecessor.get(current_configuration_index, None)
            
            if next_configuration_index == None: 
                return np.empty(shape = (0,0))
            
            path.append(next_configuration_index)
            current_configuration = self.vertices[next_configuration_index]
            current_configuration_index = next_configuration_index
        
        path = np.flip(np.array(path))
        self.rrt_path = path     
        return path       
        
    #Return true if two configurations are equal
    def equal(self, configurationA, configurationB):
        configurationA, configurationB = configurationA.flatten(), configurationB.flatten()
        return configurationA[0] == configurationB[0] and configurationA[1] == configurationB[1] and configurationA[2] == configurationB[2]
    
    #Function Responsible for initializing configuration
    def init_animation_configuration(self):
        configuration = self.vertices[self.rrt_path[0]]
        rigid_body = self.rigid_body.generate_rigid_body_from_configuration(configuration)
        
        self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')
        self.rigid_body.ax.add_patch(self.patch)
        
        self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
        self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
        self.rigid_body.ax.add_line(self.path)
        
        self.body_centroid[0].set_data([configuration[0], configuration[1]])
        
        self.frame_number = 0
        return self.patch, self.path, self.body_centroid[0]
    
    #Function responsible for plotting a configuration
    def update_animation_configuration(self, frame):        
        configuration = self.vertices[self.rrt_path[frame]]
        rigid_body = self.rigid_body.generate_rigid_body_from_configuration(configuration)
        
        self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')
        self.rigid_body.ax.add_patch(self.patch)
        
        self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
        self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
        self.rigid_body.ax.add_line(self.path)
        
        self.body_centroid[0].set_data([configuration[0], configuration[1]])
        
        self.frame_number = frame
        return self.patch, self.path, self.body_centroid[0]
            
#Class Representing Probabilistic Road Map
class PRM:
    def __init__(self, N, rigid_body : RigidBody):
        #Initialize       
        self.rigid_body = rigid_body
        self.vertices = np.empty(shape = (0, 3))
        self.edges = np.empty(shape = (0,0))
        
        #Generate Roadmap
        self.generate_roadmap(N)
                
        #Set up Animation Stuff
        #Animation Object
        self.animation = None
        
        #Centroid Points
        self.centroid_points = np.empty(shape = (0, 2))
        self.frame_number = 0
    
    #Generate Roadmap
    def generate_roadmap(self, N):
        for _ in range(N):
            sampled_vertex = self.rigid_body.sample_configuration_collision_free(1)
            sampled_vertex = sampled_vertex.flatten()
            
            if len(self.vertices) == 0:
                self.vertices = np.append(self.vertices, sampled_vertex.reshape((1, sampled_vertex.shape[0])), axis = 0)
                self.edges = np.vstack((self.edges, np.zeros(shape = (1, self.edges.shape[1]))))
                self.edges = np.hstack((self.edges, np.zeros(shape = (self.edges.shape[0], 1))))
                print(self.vertices.shape, self.edges.shape)
                continue
            
            #Compute neighbor distances
            k = 3
            target_configuration = sampled_vertex
            configurations = self.vertices
            neighbor_distances = np.apply_along_axis(func1d = self.D, axis = 1, arr = configurations - target_configuration)
            neighbor_indices = np.argsort(neighbor_distances) [:k]
            
            #Add sampled vertex to self.vertices
            self.vertices = np.append(self.vertices, sampled_vertex.reshape((1, sampled_vertex.shape[0])), axis = 0)
            index = len(self.vertices) - 1
            
            #Expand self.edges
            self.edges = np.vstack((self.edges, np.zeros(shape = (1, self.edges.shape[1]))))
            self.edges = np.hstack((self.edges, np.zeros(shape = (self.edges.shape[0], 1))))
            
            #Try to add edges to neighbor nodes if possible
            for neighbor_index in neighbor_indices:
                if self.is_edge_valid(target_configuration, self.vertices[neighbor_index]):
                    self.edges[index, neighbor_index] = 1
                    self.edges[neighbor_index, index] = 1
                            
    #Tell if Edge is valid based on collision checking in workspace
    def is_edge_valid(self, vertexA, vertexB):
        vertexA = vertexA.flatten()
        vertexB = vertexB.flatten()
        
        #Check path from vertexA to vertexB
        timesteps = 100
        is_valid_path = True
        current_configuration = vertexA
        
        for timestep in range(1, timesteps + 1):
            dConfiguration = ((vertexB - vertexA) / timesteps)
            
            DeltaTheta = vertexB[2] - vertexA[2]
            if DeltaTheta > np.pi:
                DeltaTheta = DeltaTheta - (2 * np.pi)
            elif DeltaTheta < -1 * np.pi:
                DeltaTheta = (2 * np.pi) - DeltaTheta
                
            dTheta = DeltaTheta / timesteps
            dConfiguration[2] = dTheta
            
            next_configuration = current_configuration + dConfiguration
            
            #Make angle within range
            next_configuration[2] = next_configuration[2] - (2 * np.pi * (np.floor((next_configuration[2] + np.pi) / (2 * np.pi))))

            if self.rigid_body.check_rigid_body_collision(self.rigid_body.generate_rigid_body_from_configuration(next_configuration)):
                is_valid_path = False
                break
            
            current_configuration = next_configuration
        
        return is_valid_path

    #Shortest path from start -> goal
    def answer_query(self, start, goal):
        print(f"Going to Answer Query for Start: {start}, Goal: {goal}")
        k = 3
        target_configurations = [start, goal]
        for target_configuration in target_configurations:
            configurations = self.vertices
            neighbor_distances = np.apply_along_axis(func1d = self.D, axis = 1, arr = configurations - target_configuration)
            neighbor_indices = np.argsort(neighbor_distances) [:k]
            
            self.vertices = np.vstack((self.vertices, target_configuration.reshape((1, -1))))
            self.edges = np.vstack((self.edges, np.zeros(shape = (1, self.edges.shape[1]))))
            self.edges = np.hstack((self.edges, np.zeros(shape = (self.edges.shape[0], 1))))
            
            print(f"Number of Neighboring Indices: {len(neighbor_indices)}")
            
            for neighbor_index in neighbor_indices:
                if self.is_edge_valid(target_configuration, self.vertices[neighbor_index]):
                    print("VALID EDGE")
                    self.edges[-1, neighbor_index] = 1
                    self.edges[neighbor_index, -1] = 1
        
        #RUN SEARCH ALGORITHM HERE
        a_star_path_cost, a_star_path = self.A_star(start, goal)
        self.prm_path = a_star_path
        
        #Set up centroid of rigid body
        self.body_centroid = self.rigid_body.ax.plot(start[0], start[1], marker='o', markersize=3, color="green")
        
        return a_star_path_cost, a_star_path
     
    #A Star Search Algorithm.          
    def A_star(self, start, goal):
        #Set up Priority Queue
        fringe = PriorityQueue()
        
        #Set up Fringe
        start_index, goal_index = len(self.vertices) - 2, len(self.vertices) - 1
        fringe.put(item = (0 + self.H(start, goal), (start, start_index)))

        #Set up lists to determine which vertices are in the fringe and which are in the closed list
        in_fringe = np.zeros(shape = (len(self.vertices),))
        in_fringe[start_index] = 1
        closed = np.zeros(shape = (len(self.vertices),))
        
        #Initialize variable to keep track of total optimal path cost
        a_star_path_cost = None
        
        ## Set up parents list
        parents = np.full(shape = (len(self.vertices),), fill_value = -1)
        
        #Iterate while fringe is NOT empty
        while not fringe.empty():
            node = fringe.get() #Get Node from Fringe
            F_value, node_configuration, node_index = node[0], node[1][0], node[1][1]
            
            closed[node_index] = 1 #Mark node as closed
            in_fringe[node_index] = 0 #Mark node as out of fringe
            
            #If we have reached goal, we are done!
            if node_index == goal_index:
                a_star_path_cost = F_value
                print("Found the goal Node with a cost of", a_star_path_cost)
                break
            
            #Calculate G Value for Node
            G_value_node = F_value - self.H(node_configuration, goal)
            
            #Iterate through edges of node to find children to add to fringe
            for child_index in range(len(self.vertices)):
                if self.edges[node_index][child_index] == 0 or closed[child_index] == 1:
                    continue
                
                child_configuration = self.vertices[child_index]
                if in_fringe[child_index] == 1:
                    fringe_list = [] #Initialize List
                    
                    #Populate fringe list with Priority Queue elements
                    while not fringe.empty():
                        fringe_list.append(fringe.get())
                       
                    #Update 
                    for i, fringe_node in enumerate(fringe_list):
                        fringe_node_F_value, fringe_node_configuration, fringe_node_index = fringe_node[0], fringe_node[1][0], fringe_node[1][1]
                        if fringe_node_index == child_index:
                            new_G_value = G_value_node + self.D(child_configuration - node_configuration)
                            child_H_value = self.H(child_configuration, goal)
                            if new_G_value +  child_H_value < fringe_node_F_value:
                                fringe_list[i] = (new_G_value + child_H_value, (child_configuration, child_index))
                                parents[child_index] = node_index
                    
                    #Convert back to heap
                    fringe = PriorityQueue()
                    for item in fringe_list:
                        fringe.put(item)
                    
                else:
                    fringe.put(item = (G_value_node + self.D(child_configuration - node_configuration) + self.H(child_configuration, goal), (child_configuration, child_index)))
                    in_fringe[child_index] = 1
                    parents[child_index] = node_index
        
        #Retrieve the actual A* Path
        current_node = goal_index
        a_star_path = [goal_index]
        while current_node != start_index:
            next_index = parents[current_node]
            if next_index == -1:
                a_star_path = []
                break
            
            a_star_path.append(next_index)
            current_node = next_index  
        
        a_star_path = np.flip(np.array(a_star_path))      
        return a_star_path_cost, a_star_path
        
    #Heuristic Function for a Node in Configuration Space
    def H(self, node, goal):
        point = goal - node
        
        #Flatten
        point = point.flatten()
        
        #Calculate Euclidean Distance Transitionally
        dt = np.linalg.norm(point[:-1])
        
        #Calculate Rotational Distance
        angle = point[-1]
        dr = min(abs(angle), 2 * np.pi - abs(angle))
        
        return 0.7 * dt + 0.3 * dr
    
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
    
    #Function Responsible for Initializing Configuration
    def init_animation_configuration(self):
        configuration = self.vertices[self.prm_path[0]]
        rigid_body = self.rigid_body.generate_rigid_body_from_configuration(configuration)
        
        self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')
        self.rigid_body.ax.add_patch(self.patch)
        
        self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
        self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
        self.rigid_body.ax.add_line(self.path)
        
        self.body_centroid[0].set_data([configuration[0], configuration[1]])
        
        self.frame_number = 0
        return self.patch, self.path, self.body_centroid[0]
    
    #Function responsible for plotting a configuration
    def update_animation_configuration(self, frame):        
        configuration = self.vertices[self.prm_path[frame]]
        rigid_body = self.rigid_body.generate_rigid_body_from_configuration(configuration)
        
        self.patch = matplotlib.patches.Polygon(rigid_body, closed=True, facecolor = 'none', edgecolor='r')
        self.rigid_body.ax.add_patch(self.patch)
        
        self.centroid_points = np.vstack((self.centroid_points, configuration[:2]))
        self.path = Line2D(self.centroid_points[:, 0].flatten(), self.centroid_points[:, 1].flatten())
        self.rigid_body.ax.add_line(self.path)
        
        self.body_centroid[0].set_data([configuration[0], configuration[1]])
        
        self.frame_number = frame
        return self.patch, self.path, self.body_centroid[0]