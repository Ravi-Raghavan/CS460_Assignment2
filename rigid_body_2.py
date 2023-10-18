### Problem 2: Motion Planning for a Rigid Body
## 2.2: Nearest neighbors with linear search approach
from utils import *

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

def D(point):
    point = point.flatten()
    return 0.7 * np.linalg.norm(point[:-1]) + 0.3 * np.deg2rad(np.abs(point[-1]))

rigid_polygons_file = "rigid_polygons.npy"
rigid_configs_file = "rigid_configs.npy"
target_configuration = np.array([1, 1, 0])

k = 5
configurations = np.load(rigid_configs_file, allow_pickle= True)
neighbor_distances = np.apply_along_axis(func1d = D, axis = 1, arr = configurations - target_configuration)
neighbor_indices = np.argsort(neighbor_distances)[:k]

rigid_body = RigidBody(f, ax, None)
rigid_body.plot_configuration(target_configuration, color = "black")
rigid_body.plot_configuration(configurations[neighbor_indices[0]], color = "red")
rigid_body.plot_configuration(configurations[neighbor_indices[1]], color = "green")
rigid_body.plot_configuration(configurations[neighbor_indices[2]], color = "blue")

for neighbor_index in neighbor_indices[3:]:
    rigid_body.plot_configuration(configurations[neighbor_index], color = "yellow")

plt.show()