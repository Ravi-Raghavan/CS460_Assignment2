from utils_p3 import *
from matplotlib.animation import FuncAnimation

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

config = np.array([ 1.69082207 , 0.43978119 , 3.10122307])

start = np.array([0.5, 0.25, 0.05])
rigid_polygons_file = "rigid_polygons.npy"

car = Car(f, ax, rigid_polygons_file, False, start)

print(car.check_rigid_body_collision(car.generate_rigid_body_from_configuration(config)))
print(car.check_wheel_collision(car.generate_wheels_from_configuration(config)))


print(np.inf - (100))

A = np.array([[1, 2, 3], [np.inf, np.inf, np.inf]])
print(A)

F = np.array([1, 2, 3])
print(F[0: -1])

configurations = np.load("rigid_configs.npy", allow_pickle= True)
print(configurations)
