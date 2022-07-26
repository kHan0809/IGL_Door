import numpy as np
import quaternion
import math
def value_inter(v1, v2, coef):
    return v1*(1-coef) + v2*(coef)


def traj_interpolation_stack(traj1,traj2,obj_path1,obj_path2,sub_goal,coef):
    traj1_len = len(traj1) - 1
    traj2_len = len(traj2) - 1
    inter_traj_real_len = (traj1_len)*(1-coef) + (traj2_len)*(coef)
    inter_traj_len = math.ceil(inter_traj_real_len)

    ratio1 = traj1_len / inter_traj_len;
    ratio2 = traj2_len / inter_traj_len;

    robot_quat1 = np.concatenate((traj1[:,6].reshape(-1,1),traj1[:, 3:6]),axis=1)
    robot_quat2 = np.concatenate((traj2[:,6].reshape(-1,1),traj2[:, 3:6]),axis=1)



    robot_q1 = quaternion.as_quat_array(robot_quat1)
    robot_q2 = quaternion.as_quat_array(robot_quat2)

    #===================================
    obj_obj1_q1 = np.concatenate((obj_path1[:,6].reshape(-1,1),obj_path1[:, 3:6]),axis=1)
    obj_obj2_q1 = np.concatenate((obj_path1[:,-1].reshape(-1,1),obj_path1[:, 10:-1]),axis=1)
    obj_o1_q1 = quaternion.as_quat_array(obj_obj1_q1)
    obj_o2_q1 = quaternion.as_quat_array(obj_obj2_q1)

    obj_obj1_q2 = np.concatenate((obj_path2[:,6].reshape(-1,1),obj_path2[:, 3:6]),axis=1)
    obj_obj2_q2 = np.concatenate((obj_path2[:,-1].reshape(-1,1),obj_path2[:, 10:-1]),axis=1)
    obj_o1_q2 = quaternion.as_quat_array(obj_obj1_q2)
    obj_o2_q2 = quaternion.as_quat_array(obj_obj2_q2)


    new_traj = []
    new_obj  = []
    for i in range(inter_traj_len+1):
        temp = []
        temp_obj = []
        if i == 0:
            # ====================robot===============
            temp.append(value_inter(traj1[0, 0], traj2[0, 0], coef))
            temp.append(value_inter(traj1[0, 1], traj2[0, 1], coef))
            temp.append(value_inter(traj1[0, 2], traj2[0, 2], coef))
            new_q = quaternion.slerp_evaluate(robot_q1[0], robot_q2[0], coef)
            temp.extend(quaternion.as_float_array(new_q))
            temp.append(value_inter(traj1[0, 7], traj2[0, 7], coef))
            temp.append(value_inter(traj1[0, 8], traj2[0, 8], coef))
            #====================obj1===============
            temp_obj.append(value_inter(obj_path1[0, 0], obj_path2[0, 0], coef))
            temp_obj.append(value_inter(obj_path1[0, 1], obj_path2[0, 1], coef))
            temp_obj.append(value_inter(obj_path1[0, 2], obj_path2[0, 2], coef))
            new_o1_q = quaternion.slerp_evaluate(obj_o1_q1[0], obj_o1_q2[0], coef)
            temp_obj.extend(quaternion.as_float_array(new_o1_q))
            # ====================obj2===============
            temp_obj.append(value_inter(obj_path1[0, 7], obj_path2[0, 7], coef))
            temp_obj.append(value_inter(obj_path1[0, 8], obj_path2[0, 8], coef))
            temp_obj.append(value_inter(obj_path1[0, 9], obj_path2[0, 9], coef))
            new_o2_q = quaternion.slerp_evaluate(obj_o2_q1[0], obj_o2_q2[0], coef)
            temp_obj.extend(quaternion.as_float_array(new_o2_q))


        elif i == inter_traj_len:
            temp.append(value_inter(traj1[-1, 0], traj2[-1, 0], coef))
            temp.append(value_inter(traj1[-1, 1], traj2[-1, 1], coef))
            temp.append(value_inter(traj1[-1, 2], traj2[-1, 2], coef))
            new_q = quaternion.slerp_evaluate(robot_q1[-1], robot_q2[-1], coef)
            temp.extend(quaternion.as_float_array(new_q))
            temp.append(value_inter(traj1[-1, 7], traj2[-1, 7], coef))
            temp.append(value_inter(traj1[-1, 8], traj2[-1, 8], coef))
            #====================obj1===============
            temp_obj.append(value_inter(obj_path1[-1, 0], obj_path2[-1, 0], coef))
            temp_obj.append(value_inter(obj_path1[-1, 1], obj_path2[-1, 1], coef))
            temp_obj.append(value_inter(obj_path1[-1, 2], obj_path2[-1, 2], coef))
            new_o1_q = quaternion.slerp_evaluate(obj_o1_q1[-1], obj_o1_q2[-1], coef)
            temp_obj.extend(quaternion.as_float_array(new_o1_q))
            # ====================obj2===============
            temp_obj.append(value_inter(obj_path1[-1, 7], obj_path2[-1, 7], coef))
            temp_obj.append(value_inter(obj_path1[-1, 8], obj_path2[-1, 8], coef))
            temp_obj.append(value_inter(obj_path1[-1, 9], obj_path2[-1, 9], coef))
            new_o2_q = quaternion.slerp_evaluate(obj_o2_q1[-1], obj_o2_q2[-1],coef)
            temp_obj.extend(quaternion.as_float_array(new_o2_q))

        else:
            idx1 = i * ratio1
            idx2 = i * ratio2
            if (idx1%1.0) < 0.0001:
                new_x_1 = traj1[math.ceil(idx1), 0]
                new_y_1 = traj1[math.ceil(idx1), 1]
                new_z_1 = traj1[math.ceil(idx1), 2]
                new_q_1 = robot_q1[math.ceil(idx1)]
                new_grip_r_1 = traj1[math.ceil(idx1), 7]
                new_grip_l_1 = traj1[math.ceil(idx1), 8]
                #===========obj===============
                new_obj1_x1 = obj_path1[math.ceil(idx1), 0]
                new_obj1_y1 = obj_path1[math.ceil(idx1), 1]
                new_obj1_z1 = obj_path1[math.ceil(idx1), 2]
                new_obj1_q1 =  obj_o1_q1[math.ceil(idx1)]

                new_obj2_x1 = obj_path1[math.ceil(idx1), 7]
                new_obj2_y1 = obj_path1[math.ceil(idx1), 8]
                new_obj2_z1 = obj_path1[math.ceil(idx1), 9]
                new_obj2_q1 = obj_o2_q1[math.ceil(idx1)]

            else:
                pre = math.floor(idx1)
                cur = math.ceil(idx1)

                new_x_1 = value_inter(traj1[pre, 0],traj1[cur, 0],(idx1 % 1.0))
                new_y_1 = value_inter(traj1[pre, 1],traj1[cur, 1],(idx1 % 1.0))
                new_z_1 = value_inter(traj1[pre, 2],traj1[cur, 2],(idx1 % 1.0))
                new_q_1 = quaternion.slerp_evaluate(robot_q1[pre], robot_q1[cur], (idx1%1.0))
                new_grip_r_1 = value_inter(traj1[pre, 7],traj1[cur, 7],(idx1 % 1.0))
                new_grip_l_1 = value_inter(traj1[pre, 8],traj1[cur, 8],(idx1 % 1.0))

                #===========obj===============
                new_obj1_x1 = value_inter(obj_path1[pre, 0],obj_path1[cur, 0],(idx1 % 1.0))
                new_obj1_y1 = value_inter(obj_path1[pre, 1],obj_path1[cur, 1],(idx1 % 1.0))
                new_obj1_z1 = value_inter(obj_path1[pre, 2],obj_path1[cur, 2],(idx1 % 1.0))
                new_obj1_q1 = quaternion.slerp_evaluate(obj_o1_q1[pre], obj_o1_q1[cur], (idx1%1.0))

                new_obj2_x1 = value_inter(obj_path1[pre, 7],obj_path1[cur, 7],(idx1 % 1.0))
                new_obj2_y1 = value_inter(obj_path1[pre, 8],obj_path1[cur, 8],(idx1 % 1.0))
                new_obj2_z1 = value_inter(obj_path1[pre, 9],obj_path1[cur, 9],(idx1 % 1.0))
                new_obj2_q1 = quaternion.slerp_evaluate(obj_o2_q1[pre], obj_o2_q1[cur], (idx1%1.0))

            if (idx2 % 1.0) < 0.0001:
                new_x_2 = traj2[math.ceil(idx2), 0]
                new_y_2 = traj2[math.ceil(idx2), 1]
                new_z_2 = traj2[math.ceil(idx2), 2]
                new_q_2 = robot_q2[math.ceil(idx2)]
                new_grip_r_2 = traj2[math.ceil(idx2), 7]
                new_grip_l_2 = traj2[math.ceil(idx2), 8]

                #===========obj===============
                new_obj1_x2 = obj_path2[math.ceil(idx2), 0]
                new_obj1_y2 = obj_path2[math.ceil(idx2), 1]
                new_obj1_z2 = obj_path2[math.ceil(idx2), 2]
                new_obj1_q2 =  obj_o1_q2[math.ceil(idx2)]

                new_obj2_x2 = obj_path2[math.ceil(idx2), 7]
                new_obj2_y2 = obj_path2[math.ceil(idx2), 8]
                new_obj2_z2 = obj_path2[math.ceil(idx2), 9]
                new_obj2_q2 = obj_o2_q2[math.ceil(idx2)]
            else:
                pre = math.floor(idx2)
                cur = math.ceil(idx2)

                new_x_2 = value_inter(traj2[pre, 0],traj2[cur, 0],(idx2 % 1.0))
                new_y_2 = value_inter(traj2[pre, 1],traj2[cur, 1],(idx2 % 1.0))
                new_z_2 = value_inter(traj2[pre, 2],traj2[cur, 2],(idx2 % 1.0))
                new_q_2 = quaternion.slerp_evaluate(robot_q2[pre], robot_q2[cur], (idx2%1.0))
                new_grip_r_2 = value_inter(traj2[pre, 7],traj2[cur, 7],(idx2 % 1.0))
                new_grip_l_2 = value_inter(traj2[pre, 8],traj2[cur, 8],(idx2 % 1.0))

                #===========obj===============
                new_obj1_x2 = value_inter(obj_path2[pre, 0],obj_path2[cur, 0],(idx2 % 1.0))
                new_obj1_y2 = value_inter(obj_path2[pre, 1],obj_path2[cur, 1],(idx2 % 1.0))
                new_obj1_z2 = value_inter(obj_path2[pre, 2],obj_path2[cur, 2],(idx2 % 1.0))
                new_obj1_q2 = quaternion.slerp_evaluate(obj_o1_q2[pre], obj_o1_q2[cur], (idx2%1.0))

                new_obj2_x2 = value_inter(obj_path2[pre, 7],obj_path2[cur, 7],(idx2 % 1.0))
                new_obj2_y2 = value_inter(obj_path2[pre, 8],obj_path2[cur, 8],(idx2 % 1.0))
                new_obj2_z2 = value_inter(obj_path2[pre, 9],obj_path2[cur, 9],(idx2 % 1.0))
                new_obj2_q2 = quaternion.slerp_evaluate(obj_o2_q2[pre], obj_o2_q2[cur], (idx2%1.0))

            temp.append(value_inter(new_x_1,new_x_2,coef))
            temp.append(value_inter(new_y_1,new_y_2,coef))
            temp.append(value_inter(new_z_1,new_z_2,coef))
            new_q = quaternion.slerp_evaluate(new_q_1, new_q_2, coef)
            temp.extend(quaternion.as_float_array(new_q))
            temp.append(value_inter(new_grip_r_1,new_grip_r_2,coef))
            temp.append(value_inter(new_grip_l_1,new_grip_l_2,coef))

            temp_obj.append(value_inter(new_obj1_x1, new_obj1_x2, coef))
            temp_obj.append(value_inter(new_obj1_y1, new_obj1_y2, coef))
            temp_obj.append(value_inter(new_obj1_z1, new_obj1_z2, coef))


            new_obj1_q = quaternion.slerp_evaluate(new_obj1_q1, new_obj1_q2, coef)
            temp_obj.extend(quaternion.as_float_array(new_obj1_q))

            temp_obj.append(value_inter(new_obj2_x1, new_obj2_x2, coef))
            temp_obj.append(value_inter(new_obj2_y1, new_obj2_y2, coef))
            temp_obj.append(value_inter(new_obj2_z1, new_obj2_z2, coef))
            new_obj2_q = quaternion.slerp_evaluate(new_obj2_q1, new_obj2_q2, coef)
            temp_obj.extend(quaternion.as_float_array(new_obj2_q))

        new_traj.append(temp)
        new_obj.append(temp_obj)
    # raise
    new_subgoal = [sub_goal] * len(new_traj)
    new_traj = np.array(new_traj)
    new_traj = np.concatenate((new_traj[:,:3],new_traj[:,4:7],new_traj[:,3].reshape(-1,1),new_traj[:,7:]),axis=1).tolist()

    new_obj = np.array(new_obj)
    new_obj = np.concatenate((new_obj[:, :3],   new_obj[:, 4:7], new_obj[:, 3].reshape(-1, 1),\
                              new_obj[:, 7:10], new_obj[:, 11:], new_obj[:, 10].reshape(-1, 1)
                              ),axis=1).tolist()

    return dict(obs_robot=new_traj, obs_obj=new_obj, sg=new_subgoal)



def traj_interpolation_Door(traj1,traj2,obj_path1,obj_path2,sub_goal,coef):
    traj1_len = len(traj1) - 1
    traj2_len = len(traj2) - 1
    inter_traj_real_len = (traj1_len) * (1 - coef) + (traj2_len) * (coef)
    inter_traj_len = math.ceil(inter_traj_real_len)

    ratio1 = traj1_len / inter_traj_len;
    ratio2 = traj2_len / inter_traj_len;

    robot_quat1 = np.concatenate((traj1[:,6].reshape(-1,1),traj1[:, 3:6]),axis=1)
    robot_quat2 = np.concatenate((traj2[:,6].reshape(-1,1),traj2[:, 3:6]),axis=1)

    robot_q1 = quaternion.as_quat_array(robot_quat1)
    robot_q2 = quaternion.as_quat_array(robot_quat2)

    new_traj = []
    new_obj  = []
    for i in range(inter_traj_len+1):
        temp = []
        temp_obj = []
        if i == 0:
            # ====================robot===============
            temp.append(value_inter(traj1[0, 0], traj2[0, 0], coef))
            temp.append(value_inter(traj1[0, 1], traj2[0, 1], coef))
            temp.append(value_inter(traj1[0, 2], traj2[0, 2], coef))
            new_q = quaternion.slerp_evaluate(robot_q1[0], robot_q2[0], coef)
            temp.extend(quaternion.as_float_array(new_q))
            temp.append(value_inter(traj1[0, 7], traj2[0, 7], coef))
            temp.append(value_inter(traj1[0, 8], traj2[0, 8], coef))
            #====================obj1===============
            temp_obj.append(value_inter(obj_path1[0, 0], obj_path2[0, 0], coef))
            temp_obj.append(value_inter(obj_path1[0, 1], obj_path2[0, 1], coef))
            temp_obj.append(value_inter(obj_path1[0, 2], obj_path2[0, 2], coef))
            temp_obj.append(value_inter(obj_path1[0, 3], obj_path2[0, 3], coef))
            temp_obj.append(value_inter(obj_path1[0, 4], obj_path2[0, 4], coef))
            temp_obj.append(value_inter(obj_path1[0, 5], obj_path2[0, 5], coef))
            temp_obj.append(value_inter(obj_path1[0, 6], obj_path2[0, 6], coef))
            temp_obj.append(value_inter(obj_path1[0, 7], obj_path2[0, 7], coef))


        elif i == inter_traj_len:
            temp.append(value_inter(traj1[-1, 0], traj2[-1, 0], coef))
            temp.append(value_inter(traj1[-1, 1], traj2[-1, 1], coef))
            temp.append(value_inter(traj1[-1, 2], traj2[-1, 2], coef))
            new_q = quaternion.slerp_evaluate(robot_q1[-1], robot_q2[-1], coef)
            temp.extend(quaternion.as_float_array(new_q))
            temp.append(value_inter(traj1[-1, 7], traj2[-1, 7], coef))
            temp.append(value_inter(traj1[-1, 8], traj2[-1, 8], coef))
            # ====================obj1===============
            temp_obj.append(value_inter(obj_path1[-1, 0], obj_path2[-1, 0], coef))
            temp_obj.append(value_inter(obj_path1[-1, 1], obj_path2[-1, 1], coef))
            temp_obj.append(value_inter(obj_path1[-1, 2], obj_path2[-1, 2], coef))
            temp_obj.append(value_inter(obj_path1[-1, 3], obj_path2[-1, 3], coef))
            temp_obj.append(value_inter(obj_path1[-1, 4], obj_path2[-1, 4], coef))
            temp_obj.append(value_inter(obj_path1[-1, 5], obj_path2[-1, 5], coef))
            temp_obj.append(value_inter(obj_path1[-1, 6], obj_path2[-1, 6], coef))
            temp_obj.append(value_inter(obj_path1[-1, 7], obj_path2[-1, 7], coef))

        else:
            idx1 = i * ratio1
            idx2 = i * ratio2
            if (idx1%1.0) < 0.0001:
                new_x_1 = traj1[math.ceil(idx1), 0]
                new_y_1 = traj1[math.ceil(idx1), 1]
                new_z_1 = traj1[math.ceil(idx1), 2]
                new_q_1 = robot_q1[math.ceil(idx1)]
                new_grip_r_1 = traj1[math.ceil(idx1), 7]
                new_grip_l_1 = traj1[math.ceil(idx1), 8]
                #===========obj===============
                new_obj1_x1 = obj_path1[math.ceil(idx1), 0]
                new_obj1_y1 = obj_path1[math.ceil(idx1), 1]
                new_obj1_z1 = obj_path1[math.ceil(idx1), 2]
                new_obj1_q1 = obj_path1[math.ceil(idx1), 3]
                new_obj2_x1 = obj_path1[math.ceil(idx1), 4]
                new_obj2_y1 = obj_path1[math.ceil(idx1), 5]
                new_obj2_z1 = obj_path1[math.ceil(idx1), 6]
                new_obj2_q1 = obj_path1[math.ceil(idx1), 7]

            else:
                pre = math.floor(idx1)
                cur = math.ceil(idx1)

                new_x_1 = value_inter(traj1[pre, 0],traj1[cur, 0],(idx1 % 1.0))
                new_y_1 = value_inter(traj1[pre, 1],traj1[cur, 1],(idx1 % 1.0))
                new_z_1 = value_inter(traj1[pre, 2],traj1[cur, 2],(idx1 % 1.0))
                new_q_1 = quaternion.slerp_evaluate(robot_q1[pre], robot_q1[cur], (idx1%1.0))
                new_grip_r_1 = value_inter(traj1[pre, 7],traj1[cur, 7],(idx1 % 1.0))
                new_grip_l_1 = value_inter(traj1[pre, 8],traj1[cur, 8],(idx1 % 1.0))

                #===========obj===============
                new_obj1_x1 = value_inter(obj_path1[pre, 0],obj_path1[cur, 0],(idx1 % 1.0))
                new_obj1_y1 = value_inter(obj_path1[pre, 1],obj_path1[cur, 1],(idx1 % 1.0))
                new_obj1_z1 = value_inter(obj_path1[pre, 2],obj_path1[cur, 2],(idx1 % 1.0))
                new_obj1_q1 = value_inter(obj_path1[pre, 3], obj_path1[cur, 3], (idx1 % 1.0))

                new_obj2_x1 = value_inter(obj_path1[pre, 4],obj_path1[cur, 4],(idx1 % 1.0))
                new_obj2_y1 = value_inter(obj_path1[pre, 5],obj_path1[cur, 5],(idx1 % 1.0))
                new_obj2_z1 = value_inter(obj_path1[pre, 6],obj_path1[cur, 6],(idx1 % 1.0))
                new_obj2_q1 = value_inter(obj_path1[pre, 7],obj_path1[cur, 7],(idx1 % 1.0))

            if (idx2 % 1.0) < 0.0001:
                new_x_2 = traj2[math.ceil(idx2), 0]
                new_y_2 = traj2[math.ceil(idx2), 1]
                new_z_2 = traj2[math.ceil(idx2), 2]
                new_q_2 = robot_q2[math.ceil(idx2)]
                new_grip_r_2 = traj2[math.ceil(idx2), 7]
                new_grip_l_2 = traj2[math.ceil(idx2), 8]

                #===========obj===============
                new_obj1_x2 = obj_path2[math.ceil(idx2), 0]
                new_obj1_y2 = obj_path2[math.ceil(idx2), 1]
                new_obj1_z2 = obj_path2[math.ceil(idx2), 2]
                new_obj1_q2 = obj_path2[math.ceil(idx2), 3]
                new_obj2_x2 = obj_path2[math.ceil(idx2), 4]
                new_obj2_y2 = obj_path2[math.ceil(idx2), 5]
                new_obj2_z2 = obj_path2[math.ceil(idx2), 6]
                new_obj2_q2 = obj_path2[math.ceil(idx2), 7]
            else:
                pre = math.floor(idx2)
                cur = math.ceil(idx2)

                new_x_2 = value_inter(traj2[pre, 0],traj2[cur, 0],(idx2 % 1.0))
                new_y_2 = value_inter(traj2[pre, 1],traj2[cur, 1],(idx2 % 1.0))
                new_z_2 = value_inter(traj2[pre, 2],traj2[cur, 2],(idx2 % 1.0))
                new_q_2 = quaternion.slerp_evaluate(robot_q2[pre], robot_q2[cur], (idx2%1.0))
                new_grip_r_2 = value_inter(traj2[pre, 7],traj2[cur, 7],(idx2 % 1.0))
                new_grip_l_2 = value_inter(traj2[pre, 8],traj2[cur, 8],(idx2 % 1.0))

                #===========obj===============
                new_obj1_x2 = value_inter(obj_path2[pre, 0],obj_path2[cur, 0], (idx2 % 1.0))
                new_obj1_y2 = value_inter(obj_path2[pre, 1],obj_path2[cur, 1], (idx2 % 1.0))
                new_obj1_z2 = value_inter(obj_path2[pre, 2],obj_path2[cur, 2], (idx2 % 1.0))
                new_obj1_q2 = value_inter(obj_path2[pre, 3], obj_path2[cur, 3], (idx2 % 1.0))

                new_obj2_x2 = value_inter(obj_path2[pre, 4],obj_path2[cur, 4],(idx2 % 1.0))
                new_obj2_y2 = value_inter(obj_path2[pre, 5],obj_path2[cur, 5],(idx2 % 1.0))
                new_obj2_z2 = value_inter(obj_path2[pre, 6],obj_path2[cur, 6],(idx2 % 1.0))
                new_obj2_q2 = value_inter(obj_path2[pre, 7],obj_path2[cur, 7],(idx2 % 1.0))

            temp.append(value_inter(new_x_1,new_x_2,coef))
            temp.append(value_inter(new_y_1,new_y_2,coef))
            temp.append(value_inter(new_z_1,new_z_2,coef))
            new_q = quaternion.slerp_evaluate(new_q_1, new_q_2, coef)
            temp.extend(quaternion.as_float_array(new_q))
            temp.append(value_inter(new_grip_r_1,new_grip_r_2,coef))
            temp.append(value_inter(new_grip_l_1,new_grip_l_2,coef))

            temp_obj.append(value_inter(new_obj1_x1, new_obj1_x2, coef))
            temp_obj.append(value_inter(new_obj1_y1, new_obj1_y2, coef))
            temp_obj.append(value_inter(new_obj1_z1, new_obj1_z2, coef))
            temp_obj.append(value_inter(new_obj1_q1, new_obj1_q2, coef))

            temp_obj.append(value_inter(new_obj2_x1, new_obj2_x2, coef))
            temp_obj.append(value_inter(new_obj2_y1, new_obj2_y2, coef))
            temp_obj.append(value_inter(new_obj2_z1, new_obj2_z2, coef))
            temp_obj.append(value_inter(new_obj2_q1, new_obj2_q2, coef))

        new_traj.append(temp)
        new_obj.append(temp_obj)
    new_subgoal = [sub_goal] * len(new_traj)
    new_traj = np.array(new_traj)
    new_traj = np.concatenate((new_traj[:, :3], new_traj[:, 4:7], new_traj[:, 3].reshape(-1, 1), new_traj[:, 7:]),axis=1).tolist()
    return dict(obs_robot=new_traj, obs_obj=new_obj, sg=new_subgoal)



def fix_traj(new_traj, traj1,traj2,coef):
    traj1_len = len(traj1) - 1
    traj2_len = len(traj2) - 1

    traj1_path_dis = 0
    for i in range(traj1_len):
        traj1_path_dis += math.sqrt(math.pow(traj1[i+1,0]-traj1[i,0],2) + math.pow(traj1[i+1,1]-traj1[i,1],2) + math.pow(traj1[i+1,2]-traj1[i,2],2))
    traj1_path_dis /= traj1_len

    traj2_path_dis = 0
    for i in range(traj2_len):
        traj2_path_dis += math.sqrt(math.pow(traj2[i+1,0]-traj2[i,0],2) + math.pow(traj2[i+1,1]-traj2[i,1],2) + math.pow(traj2[i+1,2]-traj2[i,2],2))
    traj2_path_dis /= traj2_len

    expect_len = traj1_path_dis*coef + traj2_path_dis*(1-coef)
    fix_idx, flag, cur_idx, inc =[], False, 0, 1
    new_traj_robot = np.array(new_traj['obs_robot'])
    new_traj_obs = np.array(new_traj['obs_obj'])
    new_traj_sub = np.array(new_traj['sg'])
    while flag == False:
        if math.sqrt(math.pow(new_traj_robot[cur_idx,0]-new_traj_robot[cur_idx+inc,0],2) + \
                     math.pow(new_traj_robot[cur_idx,1]-new_traj_robot[cur_idx+inc,1],2) + \
                     math.pow(new_traj_robot[cur_idx,2]-new_traj_robot[cur_idx+inc,2],2)    ) > (0.9*expect_len):
            fix_idx.append(cur_idx)
            cur_idx += inc
            if cur_idx == (len(new_traj_robot)-1):
                flag = True
            inc = 1
        else:
            inc = inc + 1
            if (cur_idx + inc) >= (len(new_traj_robot)-1):
                flag = True
                fix_idx.append(len(new_traj_robot)-1)
    return dict(obs_robot=new_traj_robot[fix_idx, :].tolist(), obs_obj=new_traj_obs[fix_idx, :].tolist(), sg=new_traj_sub[fix_idx].tolist())