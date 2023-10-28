import numpy as np

value = (-0.91303594519) % (2 * np.pi) - np.pi

print(3 * np.pi/4)
print(f"Modulo: {value}")


arr = np.load("rigid_polygons.npy", allow_pickle = True)
print(arr)

arr = np.load("rigid_configs.npy", allow_pickle = True)
print(arr)