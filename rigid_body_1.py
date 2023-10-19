### Problem 2: Motion Planning for a Rigid Body
## 2.1: Sampling random collision-free configurations
from utils_p2 import *

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")
rigid_polygons_file = "rigid_polygons.npy"

rigid_body = RigidBody(f, ax, rigid_polygons_file)

#Generate sample_configurations
sample_configurations = rigid_body.sample_configuration_collision_free(5)

for sample_configuration in sample_configurations:
    rigid_body.plot_configuration(sample_configuration)
    
plt.show()