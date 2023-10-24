import argparse
import numpy as np
import utils_p1 as utl

# generate and store 5 radomly sampled configurations
def generate_config(arm):
    for _ in 5:
        collision_free = False
        collision = False
        # check for collisions, if collision regenerate
        while not collision_free:
            # generate random joint angles in radians
            joint1 = np.random.uniform(0,2*np.pi)
            joint2 = np.random.uniform(0,2*np.pi)
            # generate robot arm using random joint angles
            arm.update_joints([joint1,joint2])
            # if collision, regenerate 
            for polygon in map:
                    #collision check 
                    if utl.collides_optimized(arm.points ,polygon):
                        collision = True
                        break
            if not collision:  #collision-free, plot configuration
                collision_free = True
                arm.plot()
            
def main():
    # parse command line arguments for map
    parser = argparse.ArgumentParser(description='Random Collision Free Configurations')
    parser.add_argument('--map', type=str, help='map parameter')
    args = parser.parse_args()
    map = args.map

    # generate collision free configurations
    arm = utl.RobotArm(map, [0.3,0.15], [0.0, 0.0], joint_radius=0.05, link_width=0.1)
    arm.plot_polygons()
    generate_config(arm)
    
if __name__ == "__main__":
    main()
