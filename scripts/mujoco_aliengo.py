import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))

import mujoco_py
from mujoco_py import MjViewer
import numpy as np

from gait import Gait
from leg_controller import LegController
from linear_mpc_configs import LinearMpcConfig
from mpc import ModelPredictiveController
from robot_configs import AliengoConfig
from robot_data import RobotData
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


STATE_ESTIMATION = False

def reset(sim, robot_config):
    sim.reset()
    # q_pos_init = np.array([
    #     0, 0, 0.116536,
    #     1, 0, 0, 0,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77
    # ])
    q_pos_init = np.array([
        0, 0, robot_config.base_height_des,
        1, 0, 0, 0,
        0, 0.8, -1.6,
        0, 0.8, -1.6,
        0, 0.8, -1.6,
        0, 0.8, -1.6
    ])
    
    q_vel_init = np.array([
        0, 0, 0, 
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ])
    
    init_state = mujoco_py.cymj.MjSimState(
        time=0.0, 
        qpos=q_pos_init, 
        qvel=q_vel_init, 
        act=None, 
        udd_state={}
    )
    sim.set_state(init_state)

def get_true_simulation_data(sim):
    pos_base = sim.data.body_xpos[1]
    vel_base = sim.data.body_xvelp[1]
    quat_base = sim.data.sensordata[0:4]
    omega_base = sim.data.sensordata[4:7]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]
    pos_foothold = [
        sim.data.get_geom_xpos('fl_foot'),
        sim.data.get_geom_xpos('fr_foot'),
        sim.data.get_geom_xpos('rl_foot'),
        sim.data.get_geom_xpos('rr_foot')
    ]
    vel_foothold = [
        sim.data.get_geom_xvelp('fl_foot'),
        sim.data.get_geom_xvelp('fr_foot'),
        sim.data.get_geom_xvelp('rl_foot'),
        sim.data.get_geom_xvelp('rr_foot')
    ]
    pos_thigh = [
        sim.data.get_body_xpos("FL_thigh"),
        sim.data.get_body_xpos("FR_thigh"),
        sim.data.get_body_xpos("RL_thigh"),
        sim.data.get_body_xpos("RR_thigh")
    ]


    true_simulation_data = [
        pos_base, 
        vel_base, 
        quat_base, 
        omega_base, 
        pos_joint, 
        vel_joint, 
        touch_state, 
        pos_foothold, 
        vel_foothold, 
        pos_thigh
    ]
    # print(true_simulation_data)
    return true_simulation_data

def get_simulated_sensor_data(sim):
    imu_quat = sim.data.sensordata[0:4]
    imu_gyro = sim.data.sensordata[4:7]
    imu_accelerometer = sim.data.sensordata[7:10]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]
            
    simulated_sensor_data = [
        imu_quat, 
        imu_gyro, 
        imu_accelerometer, 
        pos_joint, 
        vel_joint, 
        touch_state
        ]
    # print(simulated_sensor_data)
    return simulated_sensor_data


def initialize_robot(sim, viewer, robot_config, robot_data, model, get_height_at_pos):
    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig, model, get_height_at_pos)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)
    init_gait = Gait.STANDING
    vel_base_des = [0., 0., 0.]
    
    for iter_counter in range(800):

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)

        robot_data.update(
            pos_base=data[0],
            lin_vel_base=data[1],
            quat_base=data[2],
            ang_vel_base=data[3],
            q=data[4],
            qdot=data[5]
        )

        init_gait.set_iteration(predictive_controller.iterations_between_mpc,iter_counter)
        swing_states = init_gait.get_swing_state()
        gait_table = init_gait.get_gait_table()

        predictive_controller.update_robot_state(robot_data)
        contact_forces = predictive_controller.update_mpc_if_needed(
            iter_counter, vel_base_des, 0., gait_table, solver='drake', debug=False, iter_debug=0)

        torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states)
        sim.data.ctrl[:] = torque_cmds

        sim.step()
        viewer.render()

def create_stairs(model, num_stairs, total_height, hfield_rows = 400, hfield_cols = 800,hfield_size = (12, 6, 10)):
    meter_to_index_scale = (hfield_size[0] * 2) / hfield_cols
    starting_x = 2
    stair_x_diff = (hfield_size[0] * 2 - starting_x) / num_stairs
    stair_z_diff = (total_height / num_stairs)
    height_data = np.zeros((hfield_rows, hfield_cols))
    for i in range(num_stairs):
        stair_height = i * stair_z_diff
        stair_x_start = starting_x + i * stair_x_diff
        stair_x_end = starting_x + (i + 1) * stair_x_diff
        x_start_index = (int)(stair_x_start / meter_to_index_scale)
        x_end_index = (int)(stair_x_end / meter_to_index_scale)

        height_data[:, x_start_index : x_end_index] = stair_height / hfield_size[2]

    model.hfield_data[:] = height_data.flatten()

def create_inclined_plane(model, total_width, total_height, hfield_rows = 400, hfield_cols = 800,hfield_size = (12, 6, 10)):
    meter_to_index_scale = (hfield_size[0] * 2) / hfield_cols
    starting_x = 2
    starting_index = (int)(starting_x / meter_to_index_scale)
    ending_x = starting_x + total_width
    ending_index = (int)(ending_x / meter_to_index_scale)
    height_data = np.zeros((hfield_rows, hfield_cols))
    for i in range(ending_index - starting_index):
        height = (i / (ending_index - starting_index)) * total_height
        height_data[:, i + starting_index] = height / hfield_size[2]

    model.hfield_data[:] = height_data.flatten()

def get_height_at_pos(x, y, model, hfield_rows = 400, hfield_cols = 800,hfield_size = (12, 6, 10), hfield_pos = (11, 0, 0), mean_over_area = False, mean_size = 0.5):
    meter_to_index_scale = (hfield_size[0] * 2) / hfield_cols
    point_x_displacement = x - (hfield_pos[0] - hfield_size[0])
    point_y_displacement = y - (hfield_pos[1] - hfield_size[1])
    x_index = (int)(point_x_displacement / meter_to_index_scale)
    y_index = (int)(point_y_displacement / meter_to_index_scale)
    index = y_index * hfield_cols + x_index
    z_val = 0
    if mean_over_area:
        mean_size_index = (int)(mean_size / meter_to_index_scale)
        hfield_data = np.array(model.hfield_data[:]).reshape(hfield_rows, hfield_cols)
        z_val = np.mean(hfield_data[y_index - mean_size_index : y_index + mean_size_index, x_index - mean_size_index: x_index + mean_size_index])
    else:
        z_val = model.hfield_data[index]
    return z_val * hfield_size[2] 

def get_deriv_mat(model, hfield_rows = 400, hfield_cols = 800,hfield_size = (12, 6, 10), hfield_pos = (11, 0, 0), mean_over_area = False, mean_size = 0.5):
    meter_to_index_scale = (hfield_size[0] * 2) / hfield_cols
    hfield_data = np.array(model.hfield_data[:]).reshape(hfield_rows, hfield_cols)
    hfield_row_deriv = (hfield_data[2:] - hfield_data[0:-2]) ** 2
    hfield_col_deriv = (hfield_data[:, 2:] - hfield_data[:, 0:-2]) ** 2
    hfield_deriv = np.zeros_like(hfield_data)
    hfield_deriv[1:-1] += hfield_row_deriv
    hfield_deriv[:, 1:-1] += hfield_col_deriv
    hfield_deriv = np.sqrt(hfield_deriv) / meter_to_index_scale
    plt.matshow(hfield_deriv)
    plt.show()
    return hfield_deriv

def main():
    cur_path = os.path.dirname(__file__)
    stairs = False
    mujoco_xml_path = os.path.join(cur_path, '../robot/aliengo/aliengo.xml')
    if stairs:
        mujoco_xml_path = os.path.join(cur_path, '../robot/aliengo/aliengo_stairs.xml')
    model = mujoco_py.load_model_from_path(mujoco_xml_path)
    #create_stairs(model, 80, 8)
    create_inclined_plane(model, 8, 4)
    get_deriv_mat(model)
    sim = mujoco_py.MjSim(model)
    
    viewer = MjViewer(sim)
    #viewer._scale = 0.001

    robot_config = AliengoConfig

    reset(sim, robot_config)
    sim.step()

    urdf_path = os.path.join(cur_path, '../robot/aliengo/urdf/aliengo.urdf')
    robot_data = RobotData(urdf_path, state_estimation=STATE_ESTIMATION)
    # initialize_robot(sim, viewer, robot_config, robot_data)

    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig, model, get_height_at_pos)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)

    gait = Gait.TROTTING16
    #gait_list = list(e for e in Gait)
    #gait = gait_list[0]
    swing_foot_trajs = [SwingFootTrajectoryGenerator(leg_idx, model, get_height_at_pos) for leg_idx in range(4)]

    vel_base_des = 0.3 * np.array([1.0, 0., 0.]) #np.array([1.2, 0., 0.])
    yaw_turn_rate_des = 0.

    iter_counter = 0

    gait_switch_iter = iter_counter + np.random.randint(low = 500, high = 1200)
    while True:

        #if iter_counter == gait_switch_iter:
        #    print("######################################### Just Switched ##############################################")
        #    gait_switch_iter = iter_counter + np.random.randint(low = 500, high = 1200)
        #    gait = gait_list[np.random.randint(0, len(gait_list))]

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)

        robot_data.update(
            pos_base=data[0],
            lin_vel_base=data[1],
            quat_base=data[2],
            ang_vel_base=data[3],
            q=data[4],
            qdot=data[5]
        )

        gait.set_iteration(predictive_controller.iterations_between_mpc, iter_counter)
        swing_states = gait.get_swing_state()
        gait_table = gait.get_gait_table()
        
        """
        foot_angles = np.zeros((4, 2))
        for i, foot_pos in enumerate(robot_data.pos_base_feet):
            angle_x, angle_y = get_inclination_angle(foot_pos[0], foot_pos[1], delta, model, get_height_at_pos)
            foot_angles[i]= [angle_x, angle_y] */
        """

        predictive_controller.update_robot_state(robot_data)

        contact_forces = predictive_controller.update_mpc_if_needed(iter_counter, vel_base_des, 
            yaw_turn_rate_des, gait_table, solver='drake', debug=False, iter_debug=0) 

        pos_targets_swingfeet = np.zeros((4, 3))
        vel_targets_swingfeet = np.zeros((4, 3))

        for leg_idx in range(4):
            if swing_states[leg_idx] > 0:   # leg is in swing state
                swing_foot_trajs[leg_idx].set_foot_placement(
                    robot_data, gait, vel_base_des, yaw_turn_rate_des
                )
                base_pos_base_swingfoot_des, base_vel_base_swingfoot_des = \
                    swing_foot_trajs[leg_idx].compute_traj_swingfoot(
                        robot_data, gait
                    )
                pos_targets_swingfeet[leg_idx, :] = base_pos_base_swingfoot_des
                vel_targets_swingfeet[leg_idx, :] = base_vel_base_swingfoot_des

        torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states, pos_targets_swingfeet, vel_targets_swingfeet)
        sim.data.ctrl[:] = torque_cmds

        sim.step()
        viewer.render()
        iter_counter += 1


        if iter_counter == 50000:
            sim.reset()
            reset(sim)
            iter_counter = 0
            break

        
if __name__ == '__main__':
    main()
