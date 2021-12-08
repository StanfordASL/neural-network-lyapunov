import numpy as np
import os
import torch

from pydrake.all import (AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer,
                         Simulator, SpatialForce, RigidTransform, Parser,
                         LinearQuadraticRegulator, AbstractValue,
                         LeafSystem, BasicVector, PortDataType, MultibodyPlant,
                         SpatialVelocity, ProcessModelDirectives,
                         LoadModelDirectives)
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.multibody import inverse_kinematics as ik
from pydrake.systems.primitives import LogOutput
import pydrake.solvers.mathematicalprogram as mp
from pydrake.math import RotationMatrix
from pydrake.common import GetDrakePath

from neural_network_lyapunov.examples.pole.pole import Pole

models_dir = os.path.join(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'simulation'), 'models')


def render_system_with_graphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


def add_package_paths(parser: Parser):
    parser.package_map().Add(
        "drake_manipulation_models",
        os.path.join(GetDrakePath(),
                     "manipulation/models"))

    parser.package_map().Add("iiwa_controller", models_dir)


class LyapunovController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.u_dim = 3

        self.pose_input_port = self.DeclareAbstractInputPort(
            "body_pose", AbstractValue.Make([RigidTransform()]))
        self.velocity_input_port = self.DeclareAbstractInputPort(
            "body_velocity", AbstractValue.Make([SpatialVelocity()]))

        self.ee_force = self.DeclareVectorOutputPort(
            "ee_force", BasicVector(self.u_dim), self.CalculateController)

    def CalculateController(self, context, output):
        # pole_state = self.pole_state_input_port.Eval(context)
        y = output.get_mutable_value()
        y[:] = 0


class LQRController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.ball_index = int(plant.GetBodyByName("ball").index())
        self.plate_index = int(plant.GetBodyByName("box").index())
        self.length = 0.82
        self.ms = 0.1649
        self.me = 0.2
        self.g = 9.81
        self.u_eq = np.array([0, 0, (self.ms + self.me) * self.g])
        self.u_dim = 3
        # Linearization around the upright equilibrium
        pole = Pole(self.ms, self.me, self.length)
        x0 = torch.tensor(np.zeros(7), dtype=pole.dtype)
        u0 = torch.tensor(self.u_eq, dtype=pole.dtype)
        A, B = pole.gradient(x0, u0)
        self.A = A.detach().numpy()
        self.B = B.detach().numpy()
        self.Q = np.diag([10, 10, 1, 1, 1, 10, 10])
        self.R = np.eye(3)
        self.K, _ = LinearQuadraticRegulator(
            self.A, self.B, self.Q, self.R)

        self.pose_input_port = self.DeclareAbstractInputPort(
            "body_pose", AbstractValue.Make([RigidTransform()]))
        self.velocity_input_port = self.DeclareAbstractInputPort(
            "body_velocity", AbstractValue.Make([SpatialVelocity()]))

        self.ee_force = self.DeclareVectorOutputPort(
            "ee_force", BasicVector(self.u_dim), self.CalculateController)

    def CalculateController(self, context, output):
        pose = self.pose_input_port.Eval(context)
        ball_pose = pose[self.ball_index]
        plate_pose = pose[self.plate_index]
        velocity = self.velocity_input_port.Eval(context)
        ball_velocity = velocity[self.ball_index]
        plate_velocity = velocity[self.plate_index]
        x_B, y_B, _ = ball_pose.translation()
        x_A, y_A, _ = plate_pose.translation()
        xd_B, yd_B, _ = ball_velocity.translational()
        xd_A, yd_A, zd_A = plate_velocity.translational()
        x = np.array([x_B - x_A, y_B - y_A, xd_A, yd_A,
                      zd_A, xd_B - xd_A, yd_B - yd_A])

        u = output.get_mutable_value()
        u[:] = -self.K @ x + self.u_eq


class IiwaController(LeafSystem):
    def __init__(self, plant_robot):
        LeafSystem.__init__(self)
        self.plant = plant_robot
        self.context = plant_robot.CreateDefaultContext()
        self.nq = plant_robot.num_actuators()

        self.ee_force = self.DeclareInputPort(
            "ee_force", PortDataType.kVectorValued, 3)

        self.iiwa_torque_controller = self.DeclareVectorOutputPort(
            "joint_torques", BasicVector(self.nq), self.CalculateController)

    def CalculateController(self, context, output):
        # ee_force = self.ee_force.Eval(context)
        y = output.get_mutable_value()
        y[:] = 0


def calculate_plate_position(plant_robot, plate_position, plate_rotation):
    ik_iwwa = ik.InverseKinematics(plant_robot)

    world_frame = plant_robot.world_frame()
    plate_frame = plant_robot.GetBodyByName("box").body_frame()

    p_WQ = plate_position
    p_EQ = np.zeros(3)
    p_EQ_lower_bound = p_EQ - 0.001
    p_EQ_upper_bound = p_EQ + 0.001

    ik_iwwa.AddPositionConstraint(
        frameB=world_frame,
        p_BQ=p_WQ,
        frameA=plate_frame,
        p_AQ_lower=p_EQ_lower_bound, p_AQ_upper=p_EQ_upper_bound)

    ik_iwwa.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=plate_rotation,
        frameBbar=plate_frame,
        R_BbarB=RotationMatrix.Identity(),
        theta_bound=0.0001)

    q = ik_iwwa.q()
    prog = ik_iwwa.get_mutable_prog()
    # result = mp.Solve(prog)
    for _ in range(20):
        result = mp.Solve(prog, np.array([0, 0.8, 0, -1.70, 0, -1.0, 0]))

        if result.is_success():
            break
    theta_solution = result.GetSolution(q)
    return theta_solution


def create_iiwa_controller_plant(gravity):
    """
    Creates plant that includes only the robot, used for controllers.
    :param gravity:
    :return:
    """
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    add_package_paths(parser)
    ProcessModelDirectives(
        LoadModelDirectives(os.path.join(models_dir,
                                         'iiwa_and_plate.yml')),
        plant, parser)
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    plant.Finalize()

    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant.GetFrameByName("iiwa_link_" + str(i)).index())

    return plant, link_frame_indices


def run_sim(gravity: np.array,
            f_C_W,
            time_step,
            is_visualizing=True):
    # Build diagram.
    builder = DiagramBuilder()

    # MultibodyPlant
    plant = MultibodyPlant(0)
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    add_package_paths(parser)

    ProcessModelDirectives(
        LoadModelDirectives(
            os.path.join(models_dir, 'iiwa_plate_and_pole.yml')),
        plant, parser)

    iiwa_model = plant.GetModelInstanceByName('iiwa')
    # plate_model = plant.GetModelInstanceByName('plate')
    pole_model = plant.GetModelInstanceByName('pole')

    plant.Finalize()

    # IIWA controller
    plant_robot, _ = create_iiwa_controller_plant(gravity)
    controller_iiwa = IiwaController(plant_robot)
    builder.AddSystem(controller_iiwa)
    builder.Connect(controller_iiwa.GetOutputPort("joint_torques"),
                    plant.get_actuation_input_port(iiwa_model))

    lqr_controller = LQRController(plant)
    builder.AddSystem(lqr_controller)
    builder.Connect(
        lqr_controller.get_output_port(0),
        controller_iiwa.ee_force)
    builder.Connect(
        plant.get_body_poses_output_port(),
        lqr_controller.GetInputPort("body_pose"))
    builder.Connect(
        plant.get_body_spatial_velocities_output_port(),
        lqr_controller.GetInputPort("body_velocity"))

    # meshcat visualizer
    if is_visualizing:
        ConnectMeshcatVisualizer(
            builder, scene_graph, frames_to_draw=[plant.GetBodyFrameIdOrThrow(
                plant.GetBodyByName("iiwa_link_7").index()),
                plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("box"
                                                                ).index()),
                plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("cylinder"
                                                                ).index())])

    # Logs
    iiwa_log = LogOutput(plant.get_state_output_port(iiwa_model), builder)
    iiwa_log.set_publish_period(0.001)
    diagram = builder.Build()

    # render_system_with_graphviz(diagram)

    # %% Run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()
    # context_controller = diagram.GetSubsystemContext(controller_iiwa,
    # context)
    context_plant = diagram.GetSubsystemContext(plant, context)

    # robot initial configuration by solving IK
    plate_position = np.array([0.7, 0, 0.3])
    plate_rotation = RotationMatrix.MakeYRotation(np.pi)
    q0 = calculate_plate_position(plant_robot, plate_position, plate_rotation)
    plant.SetPositions(context_plant, iiwa_model, q0)
    plant.SetPositions(context_plant, pole_model, np.array(
        [0, 0, 0, 1, 0.7, 0, 0.3 + 0.41 + 0.003]))

    # constant force on link 7.
    easf = ExternallyAppliedSpatialForce()
    easf.F_Bq_W = SpatialForce([0, 0, 0], f_C_W)
    easf.body_index = plant.GetBodyByName("iiwa_link_7").index()
    plant.get_applied_spatial_force_input_port().FixValue(
        context_plant, [easf])

    # Initialize simulator
    sim.Initialize()
    sim.set_target_realtime_rate(0)
    sim.AdvanceTo(1)

    return iiwa_log, controller_iiwa


if __name__ == '__main__':
    # force at C.
    f_C_W = np.array([0, 0, -20])
    # Stiffness matrix of the robot.
    Kp_iiwa = np.array([800., 600, 600, 600, 400, 200, 200])
    gravity = np.array([0, 0, -9.81])

    run_sim(
        gravity=gravity,
        f_C_W=f_C_W,
        time_step=1e-5)
