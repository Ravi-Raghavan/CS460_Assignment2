### Problem 2: Motion Planning for a Rigid Body
## 2.1: Sampling random collision-free configurations
from utils_p2 import RigidBody

import argparse
import matplotlib.pyplot as plt

#Set up Plot Information
f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

#Load in map of rigid polygons
parser = argparse.ArgumentParser(description="Receive command line arguments for Map of Rigid Map Polygons")
parser.add_argument("--map", nargs = 1, type = str, help = "String parameter")
args = parser.parse_args()

#Set File value
rigid_polygons_file = args.map[0] if args.map != None else args.map
print(f"Rigid Polygons File: {rigid_polygons_file}")

rigid_body = RigidBody(f, ax, rigid_polygons_file)

#Generate sample_configurations
sample_configurations = rigid_body.sample_configuration_collision_free(5)

for sample_configuration in sample_configurations:
    rigid_body.plot_configuration(sample_configuration)
    
plt.show()