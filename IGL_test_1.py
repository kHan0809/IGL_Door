import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from Model.Model import IGL, InvDyn_add, IGL_large, IGL_large_sep
import torch
from robosuite.wrappers import VisualizationWrapper

def get_current_stage(one_state,curt_subgoal):
    flag = curt_subgoal
    if 0.03<(one_state[0] - one_state[9])<0.100 and  (one_state[1] - one_state[10])<0.023 and 0.005<(one_state[2] - one_state[11])<0.05 and flag == 0:
        flag += 1
    if one_state[12]>1.25 and flag == 1:
        flag += 1
    return flag

def _flatten_obs(obs_dict, name_list):
    ob_lst = []
    for key in name_list:
        if key in obs_dict:
            ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst)

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "Door"
    # options["env_name"] = choose_environment()
    # options["env_name"] = "Stack_with_site"

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = "Panda"

    # Choose controller
    # controller_name = choose_controller()
    controller_name = "OSC_POSE"
    # controller_name = "OSC_POSITION"
    # controller_name = "IK_POSE"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)
    print(options["controller_configs"])
    print(type(options["controller_configs"]))
    # Help message to user
    print('Press "H" to show the viewer control panel.')

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    obs = env.reset()

    # obs_robot_list = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    # obs_obj_list  = ["cubeA_pos", "cubeA_quat","cubeB_pos","cubeB_quat"]

    obs_robot_list = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    obs_obj_list  = ["handle_pos", "handle_qpos","door_pos","hinge_qpos"]

    temp_list = ["robot0_eef_pos"]

    obs_robot = _flatten_obs(obs,obs_robot_list)
    obs_obj = _flatten_obs(obs,obs_obj_list)


    one_state = np.concatenate((obs_robot, obs_obj))
    current_subgoal = np.array([get_current_stage(one_state,0)])
    # current_subgoal = np.array([0])

    env.viewer.set_camera(camera_id=0)
    env = VisualizationWrapper(env)

    # Get action limits
    low, high = env.action_spec

    all_dim = 18 # 9+8+19 + 13 + 1
    robot_dim = 9
    igl = IGL_large_sep(all_dim, robot_dim, 'cpu')
    # igl = IGL_large_sep(all_dim, robot_dim, 'cpu')
    # igl.load_state_dict(torch.load('./model_save/SEP_IGL_sg0_imp000'))
    igl.load_state_dict(torch.load('./model_save/FineSEP_IGL_sg0_imp001'))


    igl.eval()
    from collections import deque

    abnormal = deque(maxlen=4)
    from scipy.spatial.transform import Rotation as R
    key = " "
    while True:
        for i in range(1000):
            one_state = np.concatenate((one_state, current_subgoal))
            next_=igl(torch.FloatTensor(one_state).unsqueeze(0))
            # action=Inv.forward(torch.FloatTensor(obs_robot).unsqueeze(0),next_)
            next = next_.squeeze(0).detach().numpy()
            action_pos = np.array([(next[0]-obs_robot[0]),(next[1]-obs_robot[1]),(next[2]-obs_robot[2])])*1
            # action_pos = (obs_obj[:3] - obs_robot[:3])

            next_r = R.from_quat(next[3:7])
            curr_r = R.from_quat(obs_robot[3:7])
            next_euler = next_r.as_euler('zyz',degrees=False)
            curr_euler = curr_r.as_euler('zyz',degrees=False)
            action_rot  = (next_euler-curr_euler)
            action_grip = np.array([next[-1]  - obs_robot[-1]])


            if action_rot[0]>2.0:
                action_rot[0] -=np.pi*2
            elif action_rot[0]<-2.0:
                action_rot[0] += np.pi * 2

            if action_rot[1]>2.0:
                action_rot[1] -=np.pi*2
            elif action_rot[1]<-2.0:
                action_rot[1] += np.pi * 2

            if action_rot[2]>2.0:
                action_rot[2] -=np.pi*2
            elif action_rot[2]<-2.0:
                action_rot[2] += np.pi * 2
            print(action_rot)
            print("===============")
            action_rot /=10
            # action_rot = np.array([0,0,0])

            action = np.concatenate((action_pos,action_rot,action_grip))

            obs, reward, done, _ = env.step(action)



            obs_robot = _flatten_obs(obs, obs_robot_list)
            obs_obj = _flatten_obs(obs, obs_obj_list)

            # print(abnormal)
            one_state = np.concatenate((obs_robot, obs_obj))
            current_subgoal = np.array([get_current_stage(one_state,current_subgoal)])

            print(current_subgoal)
            current_subgoal = np.array([0])
            env.render()
        obs = env.reset()
        obs_robot = _flatten_obs(obs, obs_robot_list)
        obs_obj = _flatten_obs(obs, obs_obj_list)


        one_state = np.concatenate((obs_robot, obs_obj))
        current_subgoal = np.array([get_current_stage(one_state,0)])

