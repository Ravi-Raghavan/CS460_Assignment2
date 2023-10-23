from utils_p3 import *

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")
rigid_polygons_file = "rigid_polygons.npy"
car = Car(f, ax, rigid_polygons_file)
plt.show()