### Problem 2: Motion Planning for a Rigid Body
## 2.3: Interpolation along the straight line in the C-space
from utils import *

f,ax = plt.subplots(dpi = 100)
ax.set_aspect("equal")

start = np.array([0.5, 0.5, 0])
goal = np.array([1.2, 1.0, 28.6])

timesteps = 5

rigid_body = RigidBody(f, ax, None)
rigid_body.plot_configuration(start, color = "red")

for timestep in range(1, timesteps + 1):
    configuration = start + ((goal - start) * timestep / timesteps)
    rigid_body.plot_configuration(configuration, color = "red")

plt.show()
