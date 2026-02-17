import rclpy
from rclpy.node import Node

from manus_ros2_msgs.msg import ManusGlove, ManusRawNode
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np





# def rotate_quaternion_z_minus_90(qx, qy, qz, qw):
#     """Rotate quaternion by -90 degrees around Z axis."""
#     rw = 0.70710678
#     rx = 0.0
#     ry = 0.0
#     rz = -0.70710678

#     # quaternion multiplication: q_new = rot * q_old
#     new_w = rw*qw - rx*qx - ry*qy - rz*qz
#     new_x = rw*qx + rx*qw + ry*qz - rz*qy
#     new_y = rw*qy - rx*qz + ry*qw + rz*qx
#     new_z = rw*qz + rx*qy - ry*qx + rz*qw

#     return new_x, new_y, new_z, new_w

def quat_mul(ax, ay, az, aw, bx, by, bz, bw):
    """Quaternion multiplication: q = a * b"""
    qx = aw*bx + ax*bw + ay*bz - az*by
    qy = aw*by - ax*bz + ay*bw + az*bx
    qz = aw*bz + ax*by - ay*bx + az*bw
    qw = aw*bw - ax*bx - ay*by - az*bz
    return qx, qy, qz, qw

def rotate_quaternion_z(qx, qy, qz, qw, degrees):
    """Rotate quaternion by degrees around Z axis (pre-multiply)."""
    rad = np.deg2rad(degrees)
    rz = np.sin(rad / 2.0)
    rw = np.cos(rad / 2.0)
    # rot = (0,0,rz,rw)
    return quat_mul(0.0, 0.0, rz, rw, qx, qy, qz, qw)

def rotate_point_z(x, y, z, degrees):
    """Rotate point by degrees around Z axis."""
    rad = np.deg2rad(degrees)
    c = np.cos(rad)
    s = np.sin(rad)
    xr = c*x - s*y
    yr = s*x + c*y
    return xr, yr, z

def rotate_quaternion_x(qx, qy, qz, qw, degrees):
    rad = np.deg2rad(degrees)
    rx = np.sin(rad / 2.0)
    rw = np.cos(rad / 2.0)
    # rot = (rx,0,0,rw)
    return quat_mul(qx, qy, qz, qw, rx, 0.0, 0.0, rw) 

def rotate_point_x(x, y, z, degrees):
    rad = np.deg2rad(degrees)
    c = np.cos(rad)
    s = np.sin(rad)
    yr = c*y - s*z
    zr = s*y + c*z
    return x, yr, zr



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

        # -------- Mount option --------
        # top  : tracker mounted above the glove (current behavior)
        # left : tracker mounted on the left of the glove -> rotate glove +90deg about Z before attaching
        self.declare_parameter("mount", "top")
        self.mount = self.get_parameter("mount").get_parameter_value().string_value.lower()
        if self.mount not in ("top", "left"):
            self.get_logger().warn(f"Unknown mount='{self.mount}', fallback to 'top'")
            self.mount = "top"
        self.get_logger().info(f"Mount mode: {self.mount}")


        self.get_logger().info("Manus Glove Visualizer with TF + Correct Right/Left Fix started.")

    def transform_pose_into_tracker_frame(self, x, y, z, qx, qy, qz, qw):
        # ----- (A) your current base fix -----
        # original:
        # y = -y
        # x_new = y ; y_new = -x ; z_new = z
        y = -y
        x_new = y
        y_new = -x
        z_new = z

        # quaternion: your current -90deg about Z
        qx2, qy2, qz2, qw2 = rotate_quaternion_z(qx, qy, qz, qw, -90.0)

        # ----- (B) mount option -----
        # if tracker is mounted on the LEFT, rotate glove +90deg about Z before attaching
        # if self.mount == "left":
        #     x_new, y_new, z_new = rotate_point_z(x_new, y_new, z_new, +90.0)
        #     qx2, qy2, qz2, qw2 = rotate_quaternion_z(qx2, qy2, qz2, qw2, +90.0)
        if self.mount == "left":

            # Step 1: already did Z rotation above
            x_new, y_new, z_new = rotate_point_z(x_new, y_new, z_new, +90.0)
            qx2, qy2, qz2, qw2 = rotate_quaternion_z(qx2, qy2, qz2, qw2, +90.0)

            # Step 2: rotate position around rotated X
            x_new, y_new, z_new = rotate_point_x(x_new, y_new, z_new, -90.0)

            # Step 3: rotate quaternion around local X
            qx2, qy2, qz2, qw2 = rotate_quaternion_x(qx2, qy2, qz2, qw2, -90.0)

        return x_new, y_new, z_new, qx2, qy2, qz2, qw2

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

        # t.transform.translation.x = palm.pose.position.x
        # t.transform.translation.y = palm.pose.position.y
        # t.transform.translation.z = palm.pose.position.z

        # t.transform.rotation = palm.pose.orientation
        x = palm.pose.position.x
        y = palm.pose.position.y
        z = palm.pose.position.z
        qx = palm.pose.orientation.x
        qy = palm.pose.orientation.y
        qz = palm.pose.orientation.z
        qw = palm.pose.orientation.w

        x2, y2, z2, qx2, qy2, qz2, qw2 = self.transform_pose_into_tracker_frame(x, y, z, qx, qy, qz, qw)

        t.transform.translation.x = x2
        t.transform.translation.y = y2
        t.transform.translation.z = z2

        t.transform.rotation.x = qx2
        t.transform.rotation.y = qy2
        t.transform.rotation.z = qz2
        t.transform.rotation.w = qw2

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

            # x = node.pose.position.x
            # y = -node.pose.position.y
            # z = node.pose.position.z

            # x_new = y
            # y_new = -x
            # z_new = z

            # new_node.pose.position.x = x_new
            # new_node.pose.position.y = y_new
            # new_node.pose.position.z = z_new

            
            # # new_node.pose.orientation.x = node.pose.orientation.x
            # # new_node.pose.orientation.y = node.pose.orientation.y
            # # new_node.pose.orientation.z = node.pose.orientation.z
            # # new_node.pose.orientation.w = node.pose.orientation.w
            # qx = node.pose.orientation.x
            # qy = node.pose.orientation.y
            # qz = node.pose.orientation.z
            # qw = node.pose.orientation.w

            # qx2, qy2, qz2, qw2 = rotate_quaternion_z_minus_90(qx, qy, qz, qw)

            # new_node.pose.orientation.x = qx2
            # new_node.pose.orientation.y = qy2
            # new_node.pose.orientation.z = qz2
            # new_node.pose.orientation.w = qw2

            x = node.pose.position.x
            y = node.pose.position.y
            z = node.pose.position.z
            qx = node.pose.orientation.x
            qy = node.pose.orientation.y
            qz = node.pose.orientation.z
            qw = node.pose.orientation.w

            x2, y2, z2, qx2, qy2, qz2, qw2 = self.transform_pose_into_tracker_frame(x, y, z, qx, qy, qz, qw)

            new_node.pose.position.x = x2
            new_node.pose.position.y = y2
            new_node.pose.position.z = z2

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

            # x_new = y
            # y_new = -x
            # z_new = z

            # p = Point(
            #     x = x_new,
            #     y = y_new,
            #     z = z_new
            # )
            x = node.pose.position.x
            y = node.pose.position.y
            z = node.pose.position.z
            # marker points don't need quaternion, pass dummy quat
            x2, y2, z2, _, _, _, _ = self.transform_pose_into_tracker_frame(x, y, z, 0.0, 0.0, 0.0, 1.0)

            p = Point(x=x2, y=y2, z=z2)

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
