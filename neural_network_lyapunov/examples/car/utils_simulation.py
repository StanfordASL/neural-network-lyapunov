import numpy as np

# Helper functions for setting up obstacle environments, simulating
# dynamics, etc.


# Parameters
def get_parameters():

    params = {}  # Initialize parameter dictionary
    # number of rays for sensor measurements for control
    params['num_rays_control'] = 30
    # number of rays for sensor measurements for data collection
    params['num_rays_data'] = 101
    params['num_rays_visualize'] = 26
    params['senseRadius'] = 100.0  # sensing radius
    params['robotRadius'] = 0.02  # 0.27  # radius of robot
    params['robotHeight'] = 0.15 / 2  # rough height of COM of robot
    params['psi_min'] = -np.pi / 3  # sensing angle minimum for control
    params['psi_max'] = np.pi / 3  # sensing angle maximum for control
    # time horizon over which to evaluate everything
    params['T_horizon'] = 2000
    params['position_error_threshold'] = 0.2
    params['dt'] = 0.01
    params['optim_iter'] = 100
    params['pred_horizon'] = 10

    # precompute vector of sensor's angles control
    params['psi_nominal'] = np.reshape(
        np.linspace(params['psi_min'], params['psi_max'],
                    params['num_rays_control']),
        (params['num_rays_control'], 1))
    # vector of sensor's angles to cover all 360 degrees range
    # for data collection
    params['psi_nominal_full'] = np.reshape(
        np.linspace(-np.pi, np.pi,
                    params['num_rays_data']), (params['num_rays_data'], 1))

    params['psi_nominal_visualize'] = np.reshape(
        np.linspace(-np.pi / 2, np.pi / 2,
                    params['num_rays_data']), (params['num_rays_data'], 1))

    return params


# Robot dynamics


def robot_update_state(state, u, plant):
    # Dubin's Car Model
    # State: [x,y,theta]
    # x: horizontal position
    # y: vertical position
    # theta: yaw_angle from lateral (positive is anti-clockwise)

    # Dynamics:
    # xdot = v*cos(theta)
    # ydot = v*sin(theta)
    # thetadot = w

    dt = 0.01

    return plant.next_pose(state, u, dt)


def generate_goal(p, goal_pos):
    linkMasses = [0.0]
    parentIdxs = [0]

    linkInertialFramePositions = [[0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    linkJointTypes = [p.JOINT_FIXED]
    linkJointAxis = [np.array([0, 0, 1])]  # [None]*numObs

    posObs = [goal_pos.tolist()]
    orientObs = [[0, 0, 0, 1]]
    visIdxs = [
        p.createVisualShape(p.GEOM_BOX,
                            halfExtents=[0.06, 0.06, 0],
                            rgbaColor=[.5, .5, .5, 1])
    ]

    goalUid = p.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        baseInertialFramePosition=[0, 0, 0],
        baseInertialFrameOrientation=[0, 0, 0, 1],
        linkMasses=linkMasses,
        linkCollisionShapeIndices=[-1],
        linkVisualShapeIndices=visIdxs,
        linkPositions=posObs,
        linkOrientations=orientObs,
        linkParentIndices=parentIdxs,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis)

    return goalUid


# Create some obstacles
def generate_obstacles(p, heightObs, robotRadius):

    # First create bounding obstacles
    x_lim = [-10.0, 10.0]
    y_lim = [-10.0, 10.0]

    areaTotal = (x_lim[1] - x_lim[0]) * (y_lim[1] - y_lim[0])
    lambda0 = .5  # intensity (ie mean density) of the Poisson process
    numObs = np.random.poisson(lambda0 * areaTotal)
    # numObs = 100 + np.random.randint(0, 21)  # 30
    # radiusObs = 0.15
    # massObs = 0
    # visualShapeId = -1
    # +3 is because we have three bounding walls
    linkMasses = [None] * (numObs)
    colIdxs = [None] * (numObs)
    visIdxs = [None] * (numObs)
    posObs = [None] * (numObs)
    orientObs = [None] * (numObs)
    parentIdxs = [None] * (numObs)

    linkInertialFramePositions = [None] * (numObs)
    linkInertialFrameOrientations = [None] * (numObs)
    linkJointTypes = [None] * (numObs)
    linkJointAxis = [None] * (numObs)

    # +3 is because we have three bounding walls
    # linkMasses = [None] * (numObs + 3)
    # colIdxs = [None] * (numObs + 3)
    # visIdxs = [None] * (numObs + 3)
    # posObs = [None] * (numObs + 3)
    # orientObs = [None] * (numObs + 3)
    # parentIdxs = [None] * (numObs + 3)
    #
    # linkInertialFramePositions = [None] * (numObs + 3)
    # linkInertialFrameOrientations = [None] * (numObs + 3)
    # linkJointTypes = [None] * (numObs + 3)
    # linkJointAxis = [None] * (numObs + 3)

    for obs in range(numObs):

        linkMasses[obs] = 0.0
        visIdxs[obs] = -1
        parentIdxs[obs] = 0

        linkInertialFramePositions[obs] = [0, 0, 0]
        linkInertialFrameOrientations[obs] = [0, 0, 0, 1]
        linkJointTypes[obs] = p.JOINT_FIXED
        linkJointAxis[obs] = np.array([0, 0, 1])  # [None]*numObs

        posObs_obs = np.array([None] * 3)
        posObs_obs[0] = x_lim[0] + \
            (x_lim[1] - x_lim[0]) * np.random.random_sample(1)
        posObs_obs[1] = 2.0 + y_lim[0] + \
            (y_lim[1] - y_lim[0] - 2.0) * \
            np.random.random_sample(1)  # Push up a bit
        posObs_obs[2] = 0  # set z at ground level
        posObs[obs] = posObs_obs  # .tolist()
        orientObs[obs] = [0, 0, 0, 1]
        colIdxs[obs] = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=(0.25 - 0.1) * np.random.random_sample(1) + 0.1,
            height=heightObs)

    # Create bounding objects
    # Left wall
    # linkMasses[numObs] = 0.0
    # visIdxs[numObs] = -1
    # parentIdxs[numObs] = 0
    # linkInertialFramePositions[numObs] = [0,0,0]
    # linkInertialFrameOrientations[numObs] = [0,0,0,1]
    # linkJointTypes[numObs] = p.JOINT_FIXED
    # linkJointAxis[numObs] = np.array([0,0,1])
    # posObs[numObs] = [x_lim[0], (y_lim[0]+y_lim[1])/2.0, 0.0]
    # orientObs[numObs] = [0,0,0,1]
    # colIdxs[numObs] = p.createCollisionShape(p.GEOM_BOX,
    # halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])
    #
    #
    # # Right wall
    # linkMasses[numObs+1] = 0.0
    # visIdxs[numObs+1] = -1
    # parentIdxs[numObs+1] = 0
    # linkInertialFramePositions[numObs+1] = [0,0,0]
    # linkInertialFrameOrientations[numObs+1] = [0,0,0,1]
    # linkJointTypes[numObs+1] = p.JOINT_FIXED
    # linkJointAxis[numObs+1] = np.array([0,0,1])
    # posObs[numObs+1] = [x_lim[1], (y_lim[0]+y_lim[1])/2.0, 0.0]
    # orientObs[numObs+1] = [0,0,0,1]
    # colIdxs[numObs+1] = p.createCollisionShape(p.GEOM_BOX,
    # halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])
    #
    # # Bottom wall
    # linkMasses[numObs+2] = 0.0
    # visIdxs[numObs+2] = -1
    # parentIdxs[numObs+2] = 0
    # linkInertialFramePositions[numObs+2] = [0,0,0]
    # linkInertialFrameOrientations[numObs+2] = [0,0,0,1]
    # linkJointTypes[numObs+2] = p.JOINT_FIXED
    # linkJointAxis[numObs+2] = np.array([0,0,1])
    # posObs[numObs+2] = [(x_lim[0]+x_lim[1])/2.0, y_lim[0], 0.0]
    # orientObs[numObs+2] = [0,0,np.sqrt(2)/2,np.sqrt(2)/2]
    # colIdxs[numObs+2] = p.createCollisionShape(p.GEOM_BOX,
    # halfExtents = [0.1, (x_lim[1] - x_lim[0])/2.0, heightObs/2])

    # # Front wall
    # linkMasses[0] = 0.0
    # visIdxs[0] = -1
    # parentIdxs[0] = 0
    # linkInertialFramePositions[0] = [0,0,0]
    # linkInertialFrameOrientations[0] = [0,0,0,1]
    # linkJointTypes[0] = p.JOINT_FIXED
    # linkJointAxis[0] = np.array([0,0,1])
    # posObs[0] = [(x_lim[0]+x_lim[1])/2.0, y_lim[1], 0.0]
    # orientObs[0] = [0,0,np.sqrt(2)/2,np.sqrt(2)/2]
    # colIdxs[0] = p.createCollisionShape(p.GEOM_BOX,
    # halfExtents = [0.1, (x_lim[1] - x_lim[0])/2.0, heightObs/2])

    obsUid = p.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        baseInertialFramePosition=[0, 0, 0],
        baseInertialFrameOrientation=[0, 0, 0, 1],
        linkMasses=linkMasses,
        linkCollisionShapeIndices=colIdxs,
        linkVisualShapeIndices=visIdxs,
        linkPositions=posObs,
        linkOrientations=orientObs,
        linkParentIndices=parentIdxs,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis)

    return obsUid


def generate_wall(p, heightObs, robotRadius):
    x_lim = [-10.0, 10.0]
    y_lim = [-10.0, 10.0]

    numObs = 1
    linkMasses = [None] * (numObs)
    colIdxs = [None] * (numObs)
    visIdxs = [None] * (numObs)
    posObs = [None] * (numObs)
    orientObs = [None] * (numObs)
    parentIdxs = [None] * (numObs)

    linkInertialFramePositions = [None] * (numObs)
    linkInertialFrameOrientations = [None] * (numObs)
    linkJointTypes = [None] * (numObs)
    linkJointAxis = [None] * (numObs)

    # # Front wall
    linkMasses[0] = 0.0
    visIdxs[0] = -1
    parentIdxs[0] = 0
    linkInertialFramePositions[0] = [0, 0, 0]
    linkInertialFrameOrientations[0] = [0, 0, 0, 1]
    linkJointTypes[0] = p.JOINT_FIXED
    linkJointAxis[0] = np.array([0, 0, 1])
    posObs[0] = [(x_lim[0] + x_lim[1]) / 2.0, y_lim[1], 0.0]
    orientObs[0] = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
    colIdxs[0] = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.1, (x_lim[1] - x_lim[0]) / 2.0, heightObs / 2])

    obsUid = p.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        baseInertialFramePosition=[0, 0, 0],
        baseInertialFrameOrientation=[0, 0, 0, 1],
        linkMasses=linkMasses,
        linkCollisionShapeIndices=colIdxs,
        linkVisualShapeIndices=visIdxs,
        linkPositions=posObs,
        linkOrientations=orientObs,
        linkParentIndices=parentIdxs,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis)

    return obsUid


# Simulate range sensor (get distances along rays)
def getDistances(p,
                 state,
                 robotHeight,
                 numRays,
                 senseRadius,
                 psi_nominal,
                 data=False,
                 visualize=False,
                 RGB=[1, 0, 0],
                 parentObjectId=None):
    """
    Get depths rays emanate from the robot
    @param p: pybullet instance
    @param senseRadius: all the distances above the senseRadius are set
    to sensRadius
    @param psi_nominal: psi angle relative to the longitudinal (+y) axis
    @param data: bool. True: depth measurements for data collection
    @return:
    """

    raysFrom = np.concatenate((state[0] * np.ones(
        (numRays, 1)), state[1] * np.ones((numRays, 1)), robotHeight * np.ones(
            (numRays, 1))), 1)

    # Note the minus sign: +ve direction for state[2] is anti-clockwise (right
    # hand rule), but sensor rays go clockwise
    thetas = (-state[2]) + psi_nominal

    raysTo = np.concatenate(
        (state[0] + senseRadius * np.cos(thetas),
         state[1] - senseRadius * np.sin(thetas), robotHeight * np.ones(
             (numRays, 1))), 1)

    coll = p.rayTestBatch(raysFrom, raysTo)

    dists = np.zeros(numRays)
    for i in range(numRays):
        dists[i] = senseRadius * coll[i][2]
        if visualize:
            if coll[i][3] == (0, 0, 0):
                p.addUserDebugLine((state[0], state[1], robotHeight),
                                   raysTo[i],
                                   lineColorRGB=RGB)
            else:
                p.addUserDebugLine((state[0], state[1], robotHeight),
                                   coll[i][3],
                                   lineColorRGB=RGB)

    if data:
        return dists[:-1]
        # not including the last ray to avoid replica (0 vs 2pi)
    else:
        return dists


# Top Down Image
def getImage(p, state, robotHeight):
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(
        [state[0], state[1], robotHeight], 10, state[2] * 180 / np.pi - 180,
        -90, 0, 2)
    proj_matrix = p.computeProjectionMatrixFOV(20, 1, 0.01, 100)
    w, h, rgba, depth, mask = p.getCameraImage(400,
                                               400,
                                               viewMatrix=viewMatrix,
                                               projectionMatrix=proj_matrix,
                                               shadow=0)
    return depth


def save_data(u_data,
              depth_data,
              state_data,
              file_name,
              next_depth_data=None,
              next_state_data=None):
    np.save('data/u_' + file_name, np.array(u_data))
    np.save('data/depth_' + file_name, np.array(depth_data))
    np.save('data/state_' + file_name, np.array(state_data))
    if next_depth_data is not None:
        np.save('data/next_depth_' + file_name, np.array(next_depth_data))
        np.save('data/next_state_' + file_name, np.array(next_state_data))
