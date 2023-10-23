from utils_p3 import *
import argparse
from matplotlib.animation import FuncAnimation


f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")
rigid_polygons_file = "rigid_polygons.npy"
car = Car(f, ax, None)

parser = argparse.ArgumentParser(description="Receive command line arguments for Car Controls")
parser.add_argument("--control", nargs=2, type=float, help="Control parameters (two floats)")
args = parser.parse_args()
control_values = args.control

if (control_values == None or len(control_values) != 2):
    print("Please Enter the correct number of arguments")
    exit(0)

v, phi = control_values
car.set_control_input(v, phi)

ani = FuncAnimation(f, car.animation_update_configuration, frames=range(1, 500), blit = True, interval=10)
plt.show()