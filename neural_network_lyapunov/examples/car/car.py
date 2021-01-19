#!/usr/bin/env python
# coding: utf-8

import pybullet as pybullet
import numpy as np
import time
import utils_simulation
import argparse
from pynput import keyboard
import unicycle

##########################################################################
# Functions


# Compute control input
def compute_optical_flow_control(y, params):
    """
    A simple controller mimicking optical flow controller to avoid obstacles.
    Assume field of view [0,pi/2)
    Make control decisions only based on views in the front.
    @param y: vector of depth measurements
    @param params:
    @return: u0 = v; u1 = w.
    """
    num_rays_control = params['num_rays_control']
    assert (num_rays_control == len(y))
    kv = .7
    kw = 1
    mid = int(num_rays_control / 2)  # assume numRays is even
    assert (2 * mid == num_rays_control)
    # angle of each ray from the longtitudinal
    assert (params['psi_max'] < np.pi / 2)
    psi = np.linspace(0, params['psi_max'], mid)
    yleft = y[:mid]
    yright = y[mid:]
    Mright = 1 / (yright * np.cos(psi))
    Mleft = 1 / (yleft * np.cos(np.flip(psi)))
    u_1 = np.array([kv, 0])
    u_1[1] = kw * (np.sum(Mright) - np.sum(Mleft))

    return u_1  # /np.linalg.norm(u_1)   #Actuation limits


def compute_optical_flow_control_goal(y,
                                      params,
                                      goal_pos=None,
                                      state=None,
                                      prev_yaw_goal=0):
    """
    Driving to the goal position while avoiding obstacles.
    Assume field of view [0,pi/2).
    Make control decisions only based on views in the front.
    Constrain the magnitude of the control [v,w] to be 1 (for less
    aggressive control law)
    @param y: vector of depth measurements
    @param params:
    @param goal_pos: goal position
    @param state: current state of the robot
    @param prev_yaw_goal: previous yaw to reach the goal position
    @return: u0 = v; u1 = w.
    """

    u_1 = compute_optical_flow_control(y, params)
    kpsi = 7.5
    yaw_goal = prev_yaw_goal
    if goal_pos is not None:
        if np.min(y) > .8:
            yaw_goal = np.arctan2(goal_pos[1] - state[1],
                                  goal_pos[0] - state[0])
            if yaw_goal < -0:
                if prev_yaw_goal > np.pi / 2:
                    yaw_goal += 2 * np.pi  # np arctan discontinuity
            u_1[1] += kpsi * (yaw_goal - state[2])
            v = np.linalg.norm(goal_pos[0:2] - state[0:2])
            if v < params['position_error_threshold']:
                v = 0
                return [0, 0]


#             u_1[0] = kv*v
    return u_1 / np.linalg.norm(u_1), yaw_goal


def user_input(u):
    """
    Generate control signals from use input.
    w: accelerate forward
    s: deccelerate
    a: turn left
    d: turn right
    @param u: control input from previous time step
    @return: current control input
    """
    k = input()
    print(k)
    if k == "w":
        u[0] += .1
    elif k == "a":
        u[1] = .5
    elif k == "s":
        u[0] -= .1
    elif k == "d":
        u[1] = -.5
    return u


def simulate_controller(numEnvs, params, husky, sphere, GUI, seed):
    """
    @param sphere: spherical area surronding the vehicle for collision
    check
    """
    # Parameters
    num_rays_control = params['num_rays_control']
    num_rays_data = params['num_rays_data']
    senseRadius = params['senseRadius']
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    psi_nominal = params['psi_nominal']
    T_horizon = params['T_horizon']

    # Fix random seed for consistency of results
    np.random.seed(seed)
    crash = 0
    reach = 0

    u_data = []
    depth_data = []
    state_data = []
    plant = unicycle.Unicycle(None)
    for env in range(0, numEnvs):
        # Sample environment
        heightObs = 20 * robotHeight
        obsUid = utils_simulation.generate_obstacles(pybullet, heightObs,
                                                     robotRadius)

        #         goal_pos = None
        while True:
            goal_pos = np.array(
                [np.random.uniform(-8, 8),
                 np.random.uniform(-8, 8), 0])
            pybullet.resetBasePositionAndOrientation(
                sphere, [goal_pos[0], goal_pos[1], robotHeight], [0, 0, 0, 1])
            # sphere and obsUid are not in collision
            if pybullet.getClosestPoints(sphere, obsUid, 0.0) == ():
                goalUid = utils_simulation.generate_goal(pybullet, goal_pos)
                break

        # # Print
        if (env % 10 == 0):
            print(env)

        # Initialize position of robot
        state = [0, 1, np.pi / 2]  # [x, y, theta]
        # pi/2 since Husky visualization is rotated by pi/2
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]])

        pybullet.resetBasePositionAndOrientation(husky,
                                                 [state[0], state[1], 0.0],
                                                 quat)
        pybullet.resetBasePositionAndOrientation(
            sphere, [state[0], state[1], robotHeight], [0, 0, 0, 1])
        yaw_goal = 0

        for t in range(0, T_horizon):
            # Get depth sensor measurement for control
            y_ray = utils_simulation.getDistances(pybullet, state, robotHeight,
                                                  num_rays_control,
                                                  senseRadius, psi_nominal)
            # Get depth sensor measurement with FOV > pi/2
            depth = utils_simulation.getDistances(pybullet, state, robotHeight,
                                                  num_rays_data, senseRadius,
                                                  params['psi_nominal_full'],
                                                  True)

            # Compute control input
            #             u = compute_optical_flow_control(y_ray,params)
            u, yaw_goal = compute_optical_flow_control_goal(
                y_ray, params, goal_pos, state, yaw_goal)
            u_data.append(u)
            depth_data.append(depth)
            state_data.append(state)

            # Update state
            state = utils_simulation.robot_update_state(state, u, plant)

            # Update position of pybullet object
            # pi/2 since Husky visualization is rotated by pi/2
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]])
            pybullet.resetBasePositionAndOrientation(husky,
                                                     [state[0], state[1], 0.0],
                                                     quat)
            pybullet.resetBasePositionAndOrientation(
                sphere, [state[0], state[1], robotHeight], [0, 0, 0, 1])

            if (GUI):
                pybullet.resetDebugVisualizerCamera(
                    cameraDistance=5.0,
                    cameraYaw=0.0,
                    cameraPitch=-45.0,
                    cameraTargetPosition=[state[0], state[1], 2 * robotHeight])
                # time.sleep(0.025)

            # Check if the robot is in collision.
            # Get closest points. Note: Last argument is distance threshold.
            # Since it's set to 0, the function will only return points if the
            # distance is less than zero. So, closestPoints is non-empty iff
            # there is a collision.
            closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)

            if closestPoints:  # Check if closestPoints is non-empty
                crash += 1
            #     break # break out of simulation for this environment

            # Check if driving out of boundary
            if state[1] > 10 or state[1] < - \
                    10 or state[0] > 10 or state[0] < -10:
                print("Terminate time:", t)
                print("Env: ", env)

            if goal_pos is not None:
                if np.linalg.norm(goal_pos[0:2] -
                                  state[0:2]) < \
                        params['position_error_threshold']:
                    pybullet.removeBody(goalUid)
                    reach += 1
                    while True:
                        goal_pos = np.array([
                            np.random.uniform(-8, 8),
                            np.random.uniform(-8, 8), 0
                        ])
                        pybullet.resetBasePositionAndOrientation(
                            sphere, [goal_pos[0], goal_pos[1], robotHeight],
                            [0, 0, 0, 1])
                        if pybullet.getClosestPoints(sphere, obsUid,
                                                     0.0) == ():
                            goalUid = utils_simulation.generate_goal(
                                pybullet, goal_pos)
                            break

        # Remove obstacles
        pybullet.removeBody(obsUid)
        pybullet.removeBody(goalUid)


#     print("Collide: "+str(crash))
#     print("Reach: "+str(reach))
    return u_data, depth_data, state_data


def simulate_random_sample(numEnvs, params, husky, sphere, GUI, seed):
    """
    @param sphere: spherical area surronding the vehicle for collision
    check
    """
    # Parameters
    num_rays_data = params['num_rays_data']
    senseRadius = params['senseRadius']
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    # psi_nominal = params['psi_nominal']
    T_horizon = params['T_horizon']

    # Fix random seed for consistency of results
    np.random.seed(seed)

    u_data = []
    depth_data = []
    next_depth_data = []
    state_data = []
    next_state_data = []
    plant = unicycle.Unicycle(None)
    for env in range(0, numEnvs):
        visualize_ray = False
        # Sample environment
        heightObs = 20 * robotHeight
        obsUid = utils_simulation.generate_obstacles(pybullet, heightObs,
                                                     robotRadius)

        # # Print
        if (env % 10 == 0):
            print(env)

        for t in range(0, T_horizon):
            # Randomly sample initial position
            while True:
                state = np.array([
                    np.random.uniform(-8, 8),
                    np.random.uniform(-8, 8),
                    np.random.uniform(-1.5 * np.pi, 1.5 * np.pi)
                ])
                pybullet.resetBasePositionAndOrientation(
                    sphere, [state[0], state[1], robotHeight], [0, 0, 0, 1])
                # sphere and obsUid are not in collision
                if pybullet.getClosestPoints(sphere, obsUid, 0.0) == ():
                    # goalUid = utils_simulation.generate_goal(pybullet,
                    #                                          goal_pos)
                    break

            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]])
            pybullet.resetBasePositionAndOrientation(husky,
                                                     [state[0], state[1], 0.0],
                                                     quat)
            pybullet.resetBasePositionAndOrientation(
                sphere, [state[0], state[1], robotHeight], [0, 0, 0, 1])

            if t == 8:
                utils_simulation.getImage(pybullet, state, robotHeight)
                time.sleep(0.1)
                visualize_ray = True

            if (GUI):
                pybullet.resetDebugVisualizerCamera(
                    cameraDistance=5.0,
                    cameraYaw=state[2] / np.pi * 180 - 120,  # 0.0,
                    cameraPitch=-89.9,  # -45.0,
                    cameraTargetPosition=[state[0], state[1], 2 * robotHeight])

            # Get depth sensor measurement with FOV > pi/2
            depth = utils_simulation.getDistances(pybullet,
                                                  state,
                                                  robotHeight,
                                                  num_rays_data,
                                                  senseRadius,
                                                  params['psi_nominal_full'],
                                                  data=True,
                                                  visualize=visualize_ray,
                                                  RGB=[1, 0, 0],
                                                  parentObjectId=husky)

            u = [np.random.uniform(-2, 5), np.random.uniform(-0.5, 0.5)]
            u_data.append(u)
            depth_data.append(depth)
            state_data.append(state)

            # Update state
            state = utils_simulation.robot_update_state(state, u, plant)

            # Update position of pybullet object
            # pi/2 since Husky visualization is rotated by pi/2
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]])
            pybullet.resetBasePositionAndOrientation(husky,
                                                     [state[0], state[1], 0.0],
                                                     quat)
            pybullet.resetBasePositionAndOrientation(
                sphere, [state[0], state[1], robotHeight], [0, 0, 0, 1])

            depth = utils_simulation.getDistances(pybullet,
                                                  state,
                                                  robotHeight,
                                                  num_rays_data,
                                                  senseRadius,
                                                  params['psi_nominal_full'],
                                                  data=True,
                                                  visualize=visualize_ray,
                                                  RGB=[0, 0, 1],
                                                  parentObjectId=husky)
            next_depth_data.append(depth)
            next_state_data.append(state)

            if t == 8:
                utils_simulation.getImage(pybullet, state, robotHeight)
                time.sleep(0.1)

            if (GUI):
                pybullet.resetDebugVisualizerCamera(
                    cameraDistance=5.0,
                    cameraYaw=state[2] / np.pi * 180 - 120,  # 0.0,
                    cameraPitch=-89.9,  # -45.0,
                    cameraTargetPosition=[state[0], state[1], 2 * robotHeight])
                # time.sleep(0.1)

        # Remove obstacles
        pybullet.removeBody(obsUid)
    return u_data, depth_data, state_data, next_depth_data, next_state_data


def press_callback(key):
    global k
    k = key.char


def continuous_user_input():
    lis = keyboard.Listener(on_press=press_callback)
    lis.start()
    lis.join()


if __name__ == "__main__":
    # Get some robot parameters
    params = utils_simulation.get_parameters()
    robotRadius = params['robotRadius']

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--file_name",
                        type=str,
                        help="file name for storing collected data")
    args = parser.parse_args()

    # Flag that sets if things are visualized
    GUI = True

    # pyBullet
    if (GUI):
        pybullet.connect(pybullet.GUI)
    else:
        pybullet.connect(pybullet.DIRECT)

    random_seed = 25
    numEnvs = 100  # Number of environments to show videos for

    # Ground plane
    pybullet.loadURDF("./URDFs/plane.urdf")

    # Load robot from URDF
    husky = pybullet.loadURDF("./URDFs/husky.urdf", globalScaling=0.5)

    # Sphere
    colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE,
                                                radius=robotRadius)
    mass = 0

    # This just makes sure that the sphere is not visible (we only use the
    # sphere for collision checking)
    visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE,
                                               radius=robotRadius,
                                               rgbaColor=[0.5, 0.5, 0.5, 0.5])

    sphere = pybullet.createMultiBody(mass, colSphereId, visualShapeId)

    print("Simulating optimized controller in a few environments...")

    # continuous_user_input()
    # Play videos
    # u_data, depth_data, state_data = \
    #     simulate_controller(numEnvs, params, husky, sphere, GUI, random_seed)
    u_data, depth_data, state_data, next_depth_data, next_state_data = \
        simulate_random_sample(numEnvs, params, husky, sphere,
                               GUI, random_seed)

    # Disconect from pybullet
    pybullet.disconnect()

    print("Done.")

    # Saving control, depth sensor and state data
    # utils_simulation.save_data(u_data, depth_data,
    #                            state_data, next_depth_data,
    #                            next_state_data, args.file_name)
