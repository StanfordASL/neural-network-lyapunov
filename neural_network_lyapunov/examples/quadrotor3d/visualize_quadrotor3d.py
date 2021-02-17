from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler
import numpy as np
import rospy
import argparse
import os


def get_line_marker(traj):
    line_marker = Marker()
    line_marker.header.frame_id = "/map"
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.color.a = .9
    line_marker.color.r = 1.0
    line_marker.color.g = 1.0
    line_marker.color.b = 0.0
    line_marker.scale.x = .01
    line_marker.id = 0
    line_points = []
    for i in range(traj.shape[1]):
        line_points.append(Point(*traj[:3, i]))
    line_marker.points = line_points
    return line_marker


def get_mesh_markers(traj, num_markers):
    mesh_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "meshes", "iris.stl")
    marker_indices = [0]
    marker_dist = int(traj.shape[1] / (num_markers - 1))
    for k in range(num_markers - 2):
        marker_indices.append(marker_indices[-1] + marker_dist)
    marker_indices.append(traj.shape[1] - 1)
    mesh_markers = []
    alphas = np.linspace(.7, .99, len(marker_indices))
    for i in range(len(marker_indices)):
        m = Marker()
        m.header.frame_id = "/map"
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = "file://" + mesh_path
        m.action = Marker.ADD
        m.color.a = alphas[i]
        m.color.r = 0.2
        m.color.g = 0.1
        m.color.b = 0.9
        m.scale.x = 1.
        m.scale.y = 1.
        m.scale.z = 1.
        m.pose.position.x = traj[0, marker_indices[i]]
        m.pose.position.y = traj[1, marker_indices[i]]
        m.pose.position.z = traj[2, marker_indices[i]]
        quat = quaternion_from_euler(*traj[3:6, marker_indices[i]])
        m.pose.orientation.x = quat[0]
        m.pose.orientation.y = quat[1]
        m.pose.orientation.z = quat[2]
        m.pose.orientation.w = quat[3]
        m.id = 1 + i
        mesh_markers.append(m)
    mesh_marker_arr = MarkerArray()
    mesh_marker_arr.markers = mesh_markers
    return mesh_marker_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--num_markers", default=4, type=int)
    args = parser.parse_args()

    traj = np.load(args.file)
    assert(isinstance(traj, np.ndarray))
    assert(len(traj.shape) == 2)
    assert(traj.shape[0] >= 6)
    assert(traj.shape[1] >= 2)

    line_marker = get_line_marker(traj)
    mesh_markers = get_mesh_markers(traj, args.num_markers)

    traj_line_pub = rospy.Publisher('traj_line', Marker, queue_size=10)
    traj_mesh_pub = rospy.Publisher('traj_mesh', MarkerArray, queue_size=10)
    rospy.init_node('quad_viz')

    while not rospy.is_shutdown():
        traj_line_pub.publish(line_marker)
        traj_mesh_pub.publish(mesh_markers)
        rospy.sleep(.1)
