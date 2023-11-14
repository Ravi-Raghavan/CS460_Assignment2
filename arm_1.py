import argparse
import numpy as np
import utils_p1 as utl

# generate radomly sampled configuration
def generate_config(arm):
    configuration = []
    collision_free = False
    collision = False
     # check for collisions, if collision regenerate
    while not collision_free:
        collision = False
        # generate random joint angles in radians
        joint1 = np.random.uniform(0,2*np.pi)
        joint2 = np.random.uniform(0,2*np.pi)
        # generate robot arm using random joint angles
        arm.update_joints([joint1,joint2])
        # if collision, regenerate 
        for polygon in arm.polygons:
                #collision check 
                if utl.collides_optimized(arm.points,polygon):
                    collision = True
                    break
                for joint in arm.joint_boxes:
                    if utl.collides_optimized(joint, polygon):
                        collision = True
                        break
        if not collision:
            collision_free = True
            configuration = (joint1,joint2)
        
    return configuration
            
def main():
    # parse command line arguments for map
    parser = argparse.ArgumentParser(description='Random Collision Free Configurations')
    parser.add_argument('--map', type=str, help='map parameter')
    args = parser.parse_args()
    map = args.map

    # generate collision free configurations
    lengths = [0.4,0.25]
    arm = utl.RobotArm(map, lengths, [0.0, 0.0], joint_radius=0.05, link_width=0.16611) #link_width is 1, 0.16611 makes it appear correctly
    arm.plot_polygons()
    configurations = []
    for _ in range(5):
        configurations.append(generate_config(arm))

    arm.plot_configs(configurations)
    utl.plt.show()

    '''
    To save polygons to .npy file:
        polygons_save = np.array(configurations)
        np.save('p1.2_configs', polygons_save)
    '''
    
if __name__ == "__main__":
    main()
