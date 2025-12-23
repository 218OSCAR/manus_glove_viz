import rclpy
from rclpy.node import Node

from manus_ros2_msgs.msg import ManusGlove, ManusRawNode
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np





def rotate_quaternion_z_minus_90(qx, qy, qz, qw):
    """Rotate quaternion by -90 degrees around Z axis."""
    rw = 0.70710678
    rx = 0.0
    ry = 0.0
    rz = -0.70710678

    # quaternion multiplication: q_new = rot * q_old
    new_w = rw*qw - rx*qx - ry*qy - rz*qz
    new_x = rw*qx + rx*qw + ry*qz - rz*qy
    new_y = rw*qy - rx*qz + ry*qw + rz*qx
    new_z = rw*qz + rx*qy - ry*qx + rz*qw

    return new_x, new_y, new_z, new_w

class ManusGloveVisualizer(Node):
    def __init__(self):
        super().__init__("manus_glove_visualizer")

        # Subscribers for both hands
        self.create_subscription(ManusGlove, "/manus_glove_0", self.glove_callback, 20)
        self.create_subscription(ManusGlove, "/manus_glove_1", self.glove_callback, 20)

        # RViz marker publishers
        self.nodes_pub = self.create_publisher(Marker, "/manus_glove_nodes", 20)
        self.lines_pub = self.create_publisher(Marker, "/manus_glove_lines", 20)

        # Publish a new ros2 topic to correct the manus_glove_data
        # self.glove_corrected_pub = self.create_publisher(ManusGlove, "/manus_glove_corrected", 20)
        self.glove_corrected_right_pub = self.create_publisher(ManusGlove, "/manus_glove_right_corrected", 20)
        self.glove_corrected_left_pub  = self.create_publisher(ManusGlove, "/manus_glove_left_corrected", 20)


        # TF broadcaster for both palms
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Manus Glove Visualizer with TF + Correct Right/Left Fix started.")

    # ---------------------------
    # Color coding by glove_id
    # ---------------------------
    def get_color(self, glove_id):
        return ColorRGBA(
            r=0.2 if glove_id == 0 else 1.0,
            g=0.6 if glove_id == 0 else 0.3,
            b=1.0 if glove_id == 0 else 0.3,
            a=1.0
        )

    def publish_palm_tf(self, msg: ManusGlove):

        palm = msg.raw_nodes[0]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()

        # ----------- MODIFY THIS -----------
        t.header.frame_id = "vive_ultimate_tracker"
        # -----------------------------------

        if msg.side.lower() == "right":
            t.child_frame_id = "manus/right_palm"
        else:
            t.child_frame_id = "manus/left_palm"

        t.transform.translation.x = palm.pose.position.x
        t.transform.translation.y = palm.pose.position.y
        t.transform.translation.z = palm.pose.position.z

        t.transform.rotation = palm.pose.orientation

        self.tf_broadcaster.sendTransform(t)

    def publish_corrected_glove(self, msg: ManusGlove):

        corrected = ManusGlove()

        corrected.glove_id = msg.glove_id
        corrected.side = msg.side
        corrected.raw_node_count = msg.raw_node_count
        corrected.ergonomics = msg.ergonomics
        corrected.ergonomics_count = msg.ergonomics_count
        corrected.raw_sensor_orientation = msg.raw_sensor_orientation
        corrected.raw_sensor = msg.raw_sensor
        corrected.raw_sensor_count = msg.raw_sensor_count

        corrected_nodes = []

        for node in msg.raw_nodes:

            new_node = ManusRawNode()

            new_node.node_id = node.node_id
            new_node.parent_node_id = node.parent_node_id
            new_node.joint_type = node.joint_type
            new_node.chain_type = node.chain_type

            # --- Translation transform ---
            x = node.pose.position.x
            y = -node.pose.position.y
            z = node.pose.position.z

            x_new = y
            y_new = -x
            z_new = z

            new_node.pose.position.x = x_new
            new_node.pose.position.y = y_new
            new_node.pose.position.z = z_new

            
            # new_node.pose.orientation.x = node.pose.orientation.x
            # new_node.pose.orientation.y = node.pose.orientation.y
            # new_node.pose.orientation.z = node.pose.orientation.z
            # new_node.pose.orientation.w = node.pose.orientation.w
            qx = node.pose.orientation.x
            qy = node.pose.orientation.y
            qz = node.pose.orientation.z
            qw = node.pose.orientation.w

            qx2, qy2, qz2, qw2 = rotate_quaternion_z_minus_90(qx, qy, qz, qw)

            new_node.pose.orientation.x = qx2
            new_node.pose.orientation.y = qy2
            new_node.pose.orientation.z = qz2
            new_node.pose.orientation.w = qw2


            corrected_nodes.append(new_node)

        corrected.raw_nodes = corrected_nodes

        # publish by side
        if msg.side.lower() == "right":
            self.glove_corrected_right_pub.publish(corrected)
        else:
            self.glove_corrected_left_pub.publish(corrected)





    # ---------------------------
    # Main callback
    # ---------------------------
    def glove_callback(self, msg: ManusGlove):

        # Publish TF
        self.publish_palm_tf(msg)

        self.publish_corrected_glove(msg)

        color = self.get_color(msg.glove_id)

        # ----------------------
        # Node Markers (Spheres)
        # ----------------------
        nodes = Marker()
        # Attach the points in the glove to the tracker
        # nodes.header.frame_id = "world"

        # Attach the points in the glove to the tracker
        nodes.header.frame_id = "vive_ultimate_tracker"
        nodes.header.stamp = self.get_clock().now().to_msg()
        nodes.ns = f"glove_nodes_{msg.glove_id}"
        nodes.id = msg.glove_id
        nodes.type = Marker.SPHERE_LIST
        nodes.action = Marker.ADD
        nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.01
        nodes.color = color

        # ----------------------
        # Skeleton Lines
        # ----------------------
        lines = Marker()
        # lines.header.frame_id = "world"
        lines.header.frame_id = "vive_ultimate_tracker"
        lines.header.stamp = self.get_clock().now().to_msg()
        lines.ns = f"glove_lines_{msg.glove_id}"
        lines.id = 100 + msg.glove_id
        lines.type = Marker.LINE_LIST
        lines.action = Marker.ADD
        lines.scale.x = 0.003
        lines.color = color

        node_pos = {}

        # ----------------------
        # 1) Add node positions
        # ----------------------
        for node in msg.raw_nodes:
            # ---------------------------
            # ⚠️ FIX: Change the frame from LHS to RHS
            # ---------------------------
            # p = Point(
            #     x=node.pose.position.x,
            #     y=-node.pose.position.y,  # ← KEY FIX
            #     z=node.pose.position.z,
            # )

            # ---------------------------
            # ⚠️ FIX: Make some changes to make the frame of the wrist the same as the frame showed in the STL wrist frame
            # ---------------------------
            x = node.pose.position.x
            y = -node.pose.position.y
            z = node.pose.position.z

            x_new = y
            y_new = -x
            z_new = z

            p = Point(
                x = x_new,
                y = y_new,
                z = z_new
            )

            nodes.points.append(p)
            node_pos[node.node_id] = p

        # ----------------------
        # 2) Add skeleton edges
        # ----------------------
        for node in msg.raw_nodes:
            pid = node.parent_node_id
            nid = node.node_id

            if pid in node_pos:
                lines.points.append(node_pos[pid])
                lines.points.append(node_pos[nid])

        # Publish the corrected gloves info
        # self.publish_corrected_glove(msg)


        # Publish markers
        self.nodes_pub.publish(nodes)
        self.lines_pub.publish(lines)


def main(args=None):
    rclpy.init(args=args)
    node = ManusGloveVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
