### Problem 2: Motion Planning for a Rigid Body
## 2.5: Implement a Probabilistic Road Map
from utils_p2 import *

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")
rigid_polygons_file = "rigid_polygons.npy"
rigid_body = RigidBody(f, ax, rigid_polygons_file)

start = np.array([0.5, 0.5, 0])
goal = np.array([1.2, 1.0, 28.6])

N = 1000
prm = PRM(N, rigid_body)
path_cost = prm.answer_query(start, goal)

print("Path Cost: ", path_cost)