import numpy as np
import cyber_lidar_frame

import sys
sys.path.append("/opt/nio/x86_64/cyber")
sys.path.append("/opt/nio/x86_64/lib/python/proto/")
#import cyber
from python_wrapper import cyber,cyber_time
from perception_object_pb2 import PerceptionObjects as ProtoPerceptionObjects
from point_cloud_pb2 import LidarRaw

#import ros msgs
import rospy
from std_msgs.msg import Header,String
from sensor_msgs.point_cloud2 import PointCloud2,PointField
from jsk_recognition_msgs.msg import BoundingBoxArray,BoundingBox


class_colors = {
    1:(1.0, 0., 0.),
    2:(1., 0., 1.),
    3:(0., 1., 0.),
    4:(0., 0., 1.),
    5:(0., 1., 1.),
    6:(0., 1., 1.),
}

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

class Cyber2Ros:
    def __init__(self,cyber_node):
        pc_topic = "/sensing/lidar/combined"
        self.pc_subscriber = cyber_node.create_reader(pc_topic, LidarRaw, self.pc_callback)
        self.publisher_pc = rospy.Publisher('/pointclouds',PointCloud2,queue_size=10)
        self.publisher_pc_label = rospy.Publisher('/pointclouds/label', String, queue_size=10)

        obj_topic = "/perception/torch_objects"
        self.obj_subscriber = cyber_node.create_reader(obj_topic, ProtoPerceptionObjects, self.obj_callback)
        self.publisher_percep_obj = rospy.Publisher('/perception/torch_objects/box', BoundingBoxArray, queue_size=10)
        self.publisher_percep_label = rospy.Publisher('/perception/torch_objects/label', String, queue_size=10)

        obj_topic = "/fusion/track_lidar_objects"
        self.track_lidar_subscriber = cyber_node.create_reader(obj_topic, ProtoPerceptionObjects, self.track_obj_callback)
        self.publisher_track_obj = rospy.Publisher('/fusion/track_lidar_objects', BoundingBoxArray, queue_size=10)

        self.last_pc_timestamp = 0
        self.last_sys_timestamp = 0
        self.pc_buffer = []



    def np_to_point_cloud(self,points, timestamp,parent_frame="lidar"):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions (m)
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """

        new_offset = timestamp - self.last_pc_timestamp
        self.last_pc_timestamp = timestamp

        now_sys = rospy.Time.now().to_nsec()
        new_sys_offset = now_sys - self.last_sys_timestamp
        self.last_sys_timestamp = now_sys

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]
        timestamp = rospy.Time(timestamp/1000000000)
        header = Header(frame_id=parent_frame, stamp=timestamp)
        print("pointcloud size: {}, time offset {}  {}".format(points.shape[0],new_offset,new_sys_offset/1e6))
        return PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3),
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

    def perception_obj_to_marker(self,percep_objs):
        timestamp = rospy.Time(percep_objs.time_meas/1000000000)
        bbox_array = BoundingBoxArray()
        bbox_array.header.frame_id = "lidar"
        bbox_array.header.stamp=timestamp
        for i in range(len(percep_objs.objects)):
            bbox = BoundingBox()
            obj = percep_objs.objects[i]
            if obj.position.x == 0 and obj.position.y == 0:
                continue
            bbox.header.frame_id = "lidar"
            bbox.header.stamp = timestamp
            quaternion = euler_to_quaternion(obj.heading,0, 0)
            bbox.pose.orientation.x = quaternion[0]
            bbox.pose.orientation.y = quaternion[1]
            bbox.pose.orientation.z = quaternion[2]
            bbox.pose.orientation.w = quaternion[3]
            bbox.pose.position.x = obj.position.x
            bbox.pose.position.y = obj.position.y
            bbox.pose.position.z = obj.position.z
            bbox.dimensions.x = obj.bounding_box.x
            bbox.dimensions.y = obj.bounding_box.y
            bbox.dimensions.z = obj.bounding_box.z
            bbox.label = obj.type
            # print("new bbox {} {} {} {} {} {} {}".format(bbox.pose.position.x,bbox.pose.position.y,bbox.pose.position.z,
            #                                              obj.bounding_box.x,obj.bounding_box.y,obj.bounding_box.z,obj.heading))
            bbox_array.boxes.append(bbox)
        return bbox_array

    def pc_callback(self,data):
        self.pc_buffer.append(data)

    def pc_process(self,data):
        # uint64_t, in ns
        timestamp = cyber_lidar_frame.get_time_stamp(data.raw_data)

        # numpy array,shape[num_points,3],xyz
        points_xyz = np.array(cyber_lidar_frame.point_cloud_to_array(data.raw_data))

        ros_pc_msg = self.np_to_point_cloud(points_xyz,timestamp)
        self.publisher_pc.publish(ros_pc_msg)

        label_msg = String()
        label_msg.data = 'PointCloud: {}'.format(timestamp)
        self.publisher_pc_label.publish(label_msg)

        print("publish new pc")

    def obj_callback(self,data):
        marker_array = self.perception_obj_to_marker(data)
        if len(marker_array.boxes):
            self.publisher_percep_obj.publish(marker_array)
            label_msg = String()
            label_msg.data = 'PerceptionObjs: {}'.format(data.time_meas)
            self.publisher_percep_label.publish(label_msg)
            print("publish percep obj")

    def track_obj_callback(self,data):
        marker_array = self.perception_obj_to_marker(data)
        self.publisher_track_obj.publish(marker_array)
        print("publish track obj")


def main():

    cyber.init()

    perception_node = cyber.Node("fusion")
    rospy.init_node("fusion")

    transfer = Cyber2Ros(perception_node)

    while not (cyber.is_shutdown() or rospy.is_shutdown()):
        if len(transfer.pc_buffer) > 0:
            transfer.pc_process(transfer.pc_buffer.pop(0))
        cyber_time.Duration(0.001).sleep()
        rospy.sleep(0.001)


if __name__ == '__main__':
    main()
