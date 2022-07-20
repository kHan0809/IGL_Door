import pickle
import os
import numpy as np
import math
import quaternion


def value_inter(v1, v2, coef):
    return v1*coef + v2*(1-coef)


def traj_interpolation(traj1,traj2,obj_path1,obj_path2,sub_goal,coef):
    traj1_len = len(traj1) - 1
    traj2_len = len(traj2) - 1
    inter_traj_real_len = (traj1_len)*coef + (traj2_len)*(1-coef)
    inter_traj_len = math.ceil(inter_traj_real_len)

    ratio1 = traj1_len / inter_traj_len;
    ratio2 = traj2_len / inter_traj_len;

    robot_quat1 = traj1[:, 3:7]
    robot_quat2 = traj2[:, 3:7]

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
            new_q = quaternion.slerp_evaluate(robot_q1[0], robot_q2[0], (1.0-coef))
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
            new_q = quaternion.slerp_evaluate(robot_q1[-1], robot_q2[-1], (1.0-coef))
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
                new_q_1 = quaternion.slerp_evaluate(robot_q1[pre], robot_q1[cur], (1-(idx1%1.0)))
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
                new_q_2 = quaternion.slerp_evaluate(robot_q2[pre], robot_q2[cur], (1-(idx2%1.0)))
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
            new_q = quaternion.slerp_evaluate(new_q_1, new_q_2, (1.0-coef))
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


data_concat = []
subgoal = '1'
for pickle_data in os.listdir(os.getcwd()+'/data_IGL'):
    # if 'IGL' in pickle_data:
    if 'Inter_mid_sg'+subgoal in pickle_data:
        with open('./data_IGL/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

All_traj = []
coefs = np.linspace(0,1,4,endpoint=True)
print(coefs)
print(len(data_concat))
# middle_point = len(data_concat)//2
middle_point = 1
for i in range(len(data_concat)-1):
    for j in range(i+middle_point,len(data_concat)):
        choice=np.array([i,j])

        obs_robot1 = np.array(data_concat[choice[0]]['obs_robot'])
        obs_robot2 = np.array(data_concat[choice[1]]['obs_robot'])
        obs_obj1 = np.array(data_concat[choice[0]]['obs_obj'])
        obs_obj2 = np.array(data_concat[choice[1]]['obs_obj'])

        sub_goal1 = np.array(data_concat[choice[0]]['sg'])
        sub_goal2 = np.array(data_concat[choice[1]]['sg'])



        robot_candi1 = obs_robot1
        robot_candi2 = obs_robot2
        obj_candi1   = obs_obj1
        obj_candi2   = obs_obj2

        if i == 0 and j == middle_point:
            for coef in coefs:
                fixed_traj = traj_interpolation(robot_candi1,robot_candi2,obj_candi1,obj_candi2,sub_goal1[0],coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                All_traj.append(fixed_traj)
        elif i == 0 and j != middle_point:
            for coef in coefs[1:]:
                fixed_traj = traj_interpolation(robot_candi1,robot_candi2,obj_candi1,obj_candi2,sub_goal1[0],coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                All_traj.append(fixed_traj)
        else:
            for coef in coefs[1:-1]:
                fixed_traj = traj_interpolation(robot_candi1,robot_candi2,obj_candi1,obj_candi2,sub_goal1[0],coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                All_traj.append(fixed_traj)


print(len(All_traj))
with open('./data_IGL/Inter_using_mid_sg'+subgoal+'.pickle', 'wb') as f:
    pickle.dump(All_traj, f, pickle.HIGHEST_PROTOCOL)