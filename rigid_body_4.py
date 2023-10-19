### Problem 2: Motion Planning for a Rigid Body
## 2.4: Implement Rapidly exploring random tree
from utils_p2 import *

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")
rigid_polygons_file = "rigid_polygons.npy"
rigid_body = RigidBody(f, ax, rigid_polygons_file)

start = np.array([0.5, 0.5, 0])
goal = np.array([1.2, 1.0, 28.6])

P = 0
N = 1000

rrt = RRT(start, goal, rigid_body)

while P < N:
    configuration = rigid_body.sample_configuration_collision_free(1)[0]
    rrt.add_vertex(configuration)
    P = P + 1
    
    if P % 100 == 0:
        print(f"FINISHED P = {P}")
