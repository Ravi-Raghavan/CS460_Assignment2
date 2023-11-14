import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from queue import PriorityQueue
import matplotlib.collections as coll
##############################################################################
##################### ROBOT ARM OBJECT AND FUNCTIONS #########################
##############################################################################
class RobotArm(object):
    def __init__(self, map, link_lengths, joint_angles, joint_radius=0.05, link_width=0.1):
        if map != 'None':
            self.polygons = np.load(map, allow_pickle = True)
        self.n_links = len(link_lengths)

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.joint_radius = joint_radius
        self.link_width = link_width
        self.points = np.ones((self.n_links + 1, 2))
        self.joint_boxes = []

        self.fig,self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])
        self.pat = []
        self.num_configs = 0

        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def transformation_matrix(self, theta, length):
        return np.array([
            [np.cos(theta), -np.sin(theta), length * np.cos(theta)],
            [np.sin(theta), np.cos(theta), length * np.sin(theta)],
            [0, 0, 1]
        ])
    
    # transformation matrix approach
    def update_points(self):
        point = np.array([0, 0, 1]).reshape(3, 1)
        prev_trans = np.identity(3) # Initialize as identity matrix
        for i in range(self.n_links):
            trans = self.transformation_matrix(self.joint_angles[i], self.link_lengths[i])
            prev_trans = prev_trans @ trans
            new_point = prev_trans @ point
            new_point[0, 0] += self.points[0][0]
            new_point[1, 0] += self.points[0][1]
            self.points[i + 1][0] = new_point[0, 0]
            self.points[i + 1][1] = new_point[1, 0]

        self.joint_boxes.clear()
        for i in range(self.n_links + 1):
            x,y = self.points[i]
            radius = self.joint_radius
            top_left = (x - radius, y + radius)
            top_right = (x + radius, y + radius)
            bottom_left = (x - radius, y - radius)
            bottom_right = (x + radius, y - radius)
            box = [top_right, top_left, bottom_left, bottom_right, top_right]
            self.joint_boxes.append(np.array(box))

    def draw_rectangle(self, start, end):
        """Create a rectangle from start to end with a certain width."""
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        direction = direction / length

        # Adjust the start and end points to account for the circle radius
        start_adj = start + self.joint_radius * direction
        end_adj = end - self.joint_radius * direction

        # Calculate the perpendicular direction
        perp_direction = np.array([-direction[1], direction[0]])
        half_width_vec = 0.3 * self.link_width * perp_direction

        # Calculate the 4 corners of the rectangle
        p1 = start_adj - half_width_vec
        p2 = start_adj + half_width_vec
        p3 = end_adj + half_width_vec
        p4 = end_adj - half_width_vec

        return np.array([p1, p4, p3, p2, p1])

    def plot(self, color):
        for i in range(self.n_links):
            rectangle = self.draw_rectangle(self.points[i], self.points[i + 1])
            self.ax.plot(rectangle[:, 0], rectangle[:, 1], color=color)
            self.ax.fill(rectangle[:, 0], rectangle[:, 1], color=color, alpha = 0.4) 

        for i in range(self.n_links + 1):
            circle = patches.Circle(self.points[i], radius=self.joint_radius)
            self.pat.append(circle)

        pc = coll.PatchCollection(self.pat, facecolor=color)
        self.ax.add_collection(pc)
        self.num_configs += 1
        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])
        

    def plot_configs(self, configurations):
        for config in configurations:
            self.update_joints(config)
            color = ''
            if self.num_configs == 0:
                color = 'black'
            elif self.num_configs == 1:
                color = 'firebrick'
            elif self.num_configs == 2:
                color = 'olivedrab'
            elif self.num_configs == 3:
                color = 'steelblue'
            else:
                color = 'gold'
            self.plot(color)

    def plot_polygons(self):
        # adds polygons to map
        for polygon in self.polygons:
             self.ax.add_patch(Polygon(polygon, closed = True, ec = 'black',facecolor = 'grey', alpha = 0.5))

##############################################################################
##################### COLLISION CHECKING FUNCTIONS ###########################
##############################################################################
def get_edges(polygon):
    V = polygon.shape[0]
    edges = [[polygon[i], polygon[i + 1]] for i in range(V - 1)]
    return edges

def determinant(A):
    return (A[1, 1] * A[0, 0]) - (A[0, 1] * A[1, 0])

def euclidean_distance(point1, point2):
    return np.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)

def point_on_edge(edge, point):
    point1, point2 = edge[0], edge[1]
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])
    
    #x range on the edge
    min_x, max_x = min(point1[0], point2[0]), max(point1[0], point2[0])
    
    #y range on edge    
    min_y, max_y = min(point1[1], point2[1]), max(point1[1], point2[1])
    
    b = point2[1] - (m * point2[0])
    
    inRange = min_x <= point[0] and point[0] <= max_x and min_y <= point[1] and point[1] <= max_y
    onEdge = (point[1] == b + m * point[0])
    return inRange and onEdge

def point_contained(poly, point):
    center = poly[0] #center of polar system we are using for reference
    angles = [] #store list of angles we calculate to divide the polygon into sections
    passed_full_circle = False #keep track if we pass full circle
    
    #iterate through all the vertices aside from 'center'
    for vertex in poly[1: -1]:
        relative_point = vertex - center
        phi = np.rad2deg(np.arctan2(relative_point[1], relative_point[0]))
        
        #keep value of phi between 0 and 360
        phi = phi + 360 if phi < 0 else phi
        
        #If values start to decrease, we have passed the 0 degree mark and made a full circle. In this case, simply add 360
        if (len(angles) >= 1 and phi < angles[-1]):
            phi += 360  
            passed_full_circle = True
            
        angles.append(phi)
    
    #Calculate vector from center point to the point of interest. Also calculate the angle this makes with respect to the horizontal axis emitted from our center point
    relative_point = point - center
    point_phi = np.rad2deg(np.arctan2(relative_point[1], relative_point[0]))
    
    #Adjust the angle of point_phi accordingly
    point_phi = point_phi + 360 if point_phi < 0 else point_phi #First get it within the [0, 360] range    
    point_phi = point_phi + 360 if passed_full_circle and (point_phi < angles[0] or point_phi > angles[-1]) else point_phi #Now adjust for the fact that we may have gone past the origin in our circle
    
    #Apply a binary search
    index = np.searchsorted(angles, point_phi)
    if (index == len(angles) or (index == 0 and point_phi < angles[0])):
        return False
    
    #If angle matches exactly, shortcut check
    if (point_phi == angles[index]):
        return euclidean_distance(center, point) <= euclidean_distance(center, poly[index + 1])
    
    #If not, we must do a ray, edge check
    ray = [center, point]
    edge = [poly[index], poly[index + 1]]
            
    return (not edge_intersect(ray, edge)) or point_on_edge(edge, point)
        
def edge_intersect(edge1, edge2):
    ## Linear Segments can be written using s and t parameters. Edge 1 will be expressed using s parameter and Edge 2 will be expressed using t parameter
    
    #Edge 1 x and y values. Edge 1 connects [x1, y1] to [x2, y2]
    x1, x2, y1, y2 = edge1[0][0], edge1[1][0], edge1[0][1], edge1[1][1]
    
    #Edge 2 x and y values. Edge 2 connects [x3 y3] to [x4, y4]
    x3, x4, y3, y4 = edge2[0][0], edge2[1][0], edge2[0][1], edge2[1][1]
    
    #Determinants
    Dx = determinant(np.array([[x3 - x1, x3 - x4], [y3 - y1, y3 - y4]]))
    Dy = determinant(np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]]))
    D = determinant(np.array([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]]))
    
    if D == 0:
        return Dx == 0 and Dy == 0    
    
    s = Dx / D
    t = Dy / D
    
    return 0 <= s and s <= 1 and 0 <= t and t <= 1    


def check_collision(bbox1, bbox2):
    return not (bbox1[1][0] < bbox2[0][0] or 
                bbox1[0][0] > bbox2[1][0] or 
                bbox1[1][1] < bbox2[0][1] or 
                bbox1[0][1] > bbox2[1][1])
    
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
        
        if (poly2_projections[-1] < poly1_projections[0] or poly2_projections[0] > poly1_projections[-1]):
            return False
    return True

def collides_optimized(poly1, poly2):
    bounding_boxes = [np.array([np.min(polygon, axis=0), np.max(polygon, axis=0)]) for polygon in [poly1, poly2]]
    if (check_collision(bounding_boxes[0], bounding_boxes[1])):
        return SAT(poly1, poly2)

##############################################################################
##################### ADDITIONAL FUNCTIONS ###################################
##############################################################################
def angle_difference(angle1, angle2):
    # Calculate the difference between the angles
    difference = abs(angle1 - angle2)
    
    # Ensure the result is within the [0, 2π] range
    while difference > 2 * np.pi:
        difference -= 2 * np.pi

    return difference

def normalize_angle(angle):
    normalized_angle = angle%(2*np.pi)
    return normalized_angle

# compute distance between two configurations
def compute_distance(target, neighbor): 
    joint1_distance = abs(normalize_angle(neighbor[0])-normalize_angle(target[0]))
    joint2_distance = abs(normalize_angle(neighbor[1])-normalize_angle(target[1]))

    joint1 = min(joint1_distance, 2*np.pi-joint1_distance)
    joint2 = min(joint2_distance, 2*np.pi-joint2_distance)

    return [joint1, joint2]

# compute norm of joint angle distances
def compute_norm(joints):
     return np.sqrt(joints[0]**2+joints[1]**2)

def animate(index):
    pass

######################### LINEAR SEARCH IMPLEMENTATION ######################
def linear_search(target, k, configurations):
    # store distances between configurations and target
    distances = [] # stores (distance, joint angle 1, joint angle 2)
    i = 0

    # for each configuration, compute distance to target
    for configuration in configurations:
        i =i +1
        # compute distance and add to arr ay
        angular_difference = compute_distance(target, configuration)
        distance_to_target = compute_norm(angular_difference)
        distances.append((distance_to_target, configuration))

    # sort array in ascending order based on distance
    sorted_distances = sorted(distances, key = lambda x:x[0])

    # create array of just configurations 
    closest_neighbors = []
    for i in range(k):
        _, config = sorted_distances[i]
        closest_neighbors.append(config)
    return closest_neighbors

######################### KD-TREE IMPLEMENTATION ######################
# KD-Tree construction
class Node:
    def __init__(self, point, split_dim):
        self.point = point
        self.split_dim = split_dim
        self.left = None
        self.right = None

def kd_tree(points, depth=0):
    if len(points) == 0: #if none, all nodes added to tree
        return None

    k = len(points[0]) 
    split_dim = depth % k

    points.sort(key= lambda x : x[split_dim])

    median = len(points) // 2

    node = Node(points[median], split_dim)
    node.left = kd_tree(points[:median], depth + 1)
    node.right = kd_tree(points[median + 1:], depth + 1)

    return node

# Search KD-Tree
def k_nearest_neighbors(root, query_point, k):
    pq = PriorityQueue()
    
    def search(node):
        nonlocal pq  
        if node is None:
            return
        
        ang_dist = compute_distance(query_point, node.point)
        dist = compute_norm(ang_dist)
        if len(pq.queue) < k:
            pq.put((-dist, node.point))
        else:
            if -dist > pq.queue[0][0]:
                pq.get()
                pq.put((-dist, node.point))

        # Recursive search on child nodes
        split_dim = node.split_dim
        diff = normalize_angle(query_point[split_dim]) - normalize_angle(node.point[split_dim])
        if diff <= np.pi:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        search(first)

        if len(pq.queue) < k or -abs(diff) < -pq.queue[0][0]: 
            search(second)

    search(root)
    neighbors = [point for dist, point in sorted(pq.queue, key=lambda x: x[0], reverse = True)]
    return neighbors

from collections import defaultdict 
import heapq

class Graph:
    def __init__(graph):
        graph.dict = defaultdict(list)

    def add(graph,node,adjacent_node): 
        graph.dict[node].append(adjacent_node)
        graph.dict[adjacent_node].append(node)

    def shortest_path(graph, start, end, path =[]): 
            path = path + [start] 
            if( start == end ): 
                return path 
            short_path = None
            for node in graph.dict[start]: 
                if( node not in path ): 
                    new_path = graph.shortest_path(node, end, path) 
                    if(new_path): 
                        if(not short_path or len(new_path) < len(short_path)): 
                            short_path = new_path 
            return short_path

class PRM_Graph:
    def __init__(self):
        self.dict = defaultdict(list)

    def add(self,node, new, distance):
        # Check if the node already exists in the graph
        if node in self.dict and new in self.dict:
            # Update existing node with new neighbors and distances
            self.dict[node].update({new:distance})
            self.dict[new].update({node:distance})
        elif node in self.dict and new not in self.dict:
            self.dict[node].update({new:distance})
            self.dict[new] = {node:distance}
        elif node not in self.dict and new in self.dict:
            self.dict[node] = {new:distance}
            self.dict[new].update({node:distance})
        else:
            # Add a new node with neighbors and distances
            self.dict[node] = {new:distance}
            self.dict[new] = {node:distance}
    
    def dijkstras(self, start, goal):
        # Dictionary to store the minimum distances
        distances = {}
        predecessor = {}
        unvisited = []
        track_path = []
        for node,_ in self.dict.items():
            unvisited.append(node)

        for node in unvisited:
            distances[node] = float('inf')
        distances[start] = 0

        while unvisited:
            min_distance_node = None
            for node in unvisited:
                if min_distance_node is None:
                    min_distance_node = node
                elif distances[node]<distances[min_distance_node]:
                    min_distance_node = node
            
            path = self.dict[min_distance_node].items()
            
            for child_node, weight in path:

                if weight + distances[min_distance_node] < distances[child_node]:
                    distances[child_node] = weight + distances[min_distance_node]
                    predecessor[child_node] = min_distance_node
            
            unvisited.remove(min_distance_node)
        
        currentNode = goal
        while currentNode != start:
            try:
                track_path.insert(0,currentNode)
                currentNode = predecessor[currentNode]
            except KeyError:
                return None
        
        track_path.insert(0,start)

        if distances[goal] != float('inf'):
            return track_path
        else:
            return None

def wrap(start, goal):
    # checks if wrap occurs from 0 to 2pi
    if start <= goal:
        return ((goal-start) >= np.pi, '0')
    # checks if wrap occurs from 2pi to 0
    elif start >= goal:
        return ((start-goal) >= np.pi, '2pi')