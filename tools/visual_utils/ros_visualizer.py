import numpy as np
import rospy
from std_msgs.msg import String,Header
from sensor_msgs.point_cloud2 import PointCloud2,read_points,read_points_list,PointField
from nio_msgs.msg import PerceptionObject,PerceptionObjects
from perception_object_pb2 import PerceptionObjects as ProtoPerceptionObjects
from perception_object_pb2 import PerceptionObject as ProtoPerceptionObject
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



class Visualizer(object):
    def __init__(self):
        self.publisher_bboxs = rospy.Publisher('/bboxes', BoundingBoxArray, queue_size=10)
        self.publisher_pc = rospy.Publisher('/pointclouds',PointCloud2,queue_size=10)

    def np_to_point_cloud(self,points, timestamp,parent_frame="base_link"):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions (m)
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]
        timestamp = rospy.Time(timestamp/1000000000)
        header = Header(frame_id=parent_frame, stamp=timestamp)

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
            bbox_array.boxes.append(bbox)
        return bbox_array

    def pub_pc(self,np_pc,timestamp):
        pc = self.np_to_point_cloud(np_pc,timestamp)
        self.publisher_pc.publish(pc)


    def pub_obj(self,pred_dict,stamp):
        obj_size = pred_dict.shape[0]
        is_tracked = pred_dict.shape[1] == 10
        timestamp = rospy.Time(stamp/1000000000)

        bbox_array = BoundingBoxArray()
        bbox_array.header.frame_id = "base_link"
        bbox_array.header.stamp=timestamp

        for i in range(obj_size):
            if is_tracked:
                label = int(pred_dict[i][9])
            else:
                label = int(pred_dict[i][8])

            bbox = BoundingBox()
            bbox.header.frame_id = "base_link"
            bbox.header.stamp = timestamp
            quaternion = euler_to_quaternion(pred_dict[i][3].item(),0, 0)
            bbox.pose.orientation.x = quaternion[0]
            bbox.pose.orientation.y = quaternion[1]
            bbox.pose.orientation.z = quaternion[2]
            bbox.pose.orientation.w = quaternion[3]
            bbox.pose.position.x = pred_dict[i][0]
            bbox.pose.position.y = pred_dict[i][1]
            bbox.pose.position.z = pred_dict[i][2]
            bbox.dimensions.x = pred_dict[i][4]
            bbox.dimensions.y = pred_dict[i][5]
            bbox.dimensions.z = pred_dict[i][6]
            bbox.label = label
            bbox_array.boxes.append(bbox)
        self.publisher_bboxs.publish(bbox_array)