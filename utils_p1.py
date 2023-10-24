import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

class RobotArm(object):
    def __init__(self, map, link_lengths, joint_angles, joint_radius=0.1, link_width=0.1):
        self.polygons = np.load(map, allow_pickle = True)

        self.n_links = len(link_lengths)

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.joint_radius = joint_radius
        self.link_width = link_width
        self.points = np.ones((self.n_links + 1, 2))

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])

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

    def plot(self):
        for i in range(self.n_links):
            rectangle = self.draw_rectangle(self.points[i], self.points[i + 1])
            self.ax.plot(rectangle[:, 0], rectangle[:, 1], 'orange')
            self.ax.fill(rectangle[:, 0], rectangle[:, 1], 'orange') 

        for i in range(self.n_links + 1):
            circle = patches.Circle(self.points[i], radius=self.joint_radius, facecolor='black', alpha = 0.7)
            self.ax.add_patch(circle)

        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])
       # plt.draw()
        plt.show()

    def plot_configs(self, configurations):
        for config in configurations:
            self.update_joints(config)
            self.plot()
            plt.pause(1)
        plt.show()

    def plot_polygons(self):
        # adds polygons to map
        for polygon in self.polygons:
             self.ax.add_patch(Polygon(polygon, closed = True, ec = 'black',facecolor = 'grey', alpha = 0.3))



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



        