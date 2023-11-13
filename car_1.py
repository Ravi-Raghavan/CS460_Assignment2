from utils_p3 import *
import argparse
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser(description="Receive command line arguments for Car Controls")
parser.add_argument("--startState", nargs = 3, type=float)
parser.add_argument("--control", nargs=2, type=float, help="Control parameters (two floats)")

args = parser.parse_args()
start_state = args.startState
control_values = args.control

if (control_values == None or (not isinstance(control_values, list)) or len(control_values) != 2):
    print("Please Enter the correct number of arguments")
    exit(0)

v, phi = control_values

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")
rigid_polygons_file = "others/rigid_polygons.npy"

car = None

if start_state != None and isinstance(start_state, list) and len(start_state) == 3:
    x, y, theta = start_state
    car = Car(f, ax, None, randomize_configuration = False, selected_configuration = np.array([x, y, theta]))
    car.set_state(x, y, theta)
    car.set_control_input(v, phi)
    
    ani = FuncAnimation(f, car.animation_update_configuration, frames=range(1, 500), init_func = car.init_animation_configuration, blit = True, interval=10, repeat = False)
    plt.show()
else:
    car = Car(f, ax, None, randomize_configuration = True)
    car.set_control_input(v, phi)
    ani = FuncAnimation(f, car.animation_update_configuration, frames=range(1, 500), init_func = car.init_animation_configuration, blit = True, interval=10, repeat = False)
    plt.show()