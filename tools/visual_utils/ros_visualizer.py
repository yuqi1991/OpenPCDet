import numpy as np
import rospy
from std_msgs.msg import String,Header
from sensor_msgs.point_cloud2 import PointCloud2,read_points,read_points_list,PointField
from nio_msgs.msg import PerceptionObject,PerceptionObjects
from visualization_msgs.msg import Marker, MarkerArray

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
        self.publisher_marks = rospy.Publisher('/objects', MarkerArray, queue_size=10)
        self.publisher_objs = rospy.Publisher('/perception/objects', PerceptionObjects, queue_size=10)
        self.publisher_pc = rospy.Publisher('/pointclouds',PointCloud2,queue_size=10)

    def np_to_point_cloud(self,points, timestamp,parent_frame="lidar"):
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

    def pub_pc(self,np_pc,timestamp):
        pc = self.np_to_point_cloud(np_pc,timestamp)
        self.publisher_pc.publish(pc)


    def pub_obj(self,pred_dict,stamp):
        obj_size = pred_dict.shape[0]
        is_tracked = pred_dict.shape[1] == 10
        timestamp = rospy.Time(stamp/1000000000)

        markerArray = MarkerArray()
        percep_objs = PerceptionObjects()
        percep_objs.header.frame_id = "lidar"

        percep_objs.header.stamp = timestamp

        for i in range(obj_size):
            marker = Marker()
            per_obj = PerceptionObject()
            marker.header.frame_id = "lidar"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            if is_tracked:
                marker.id = int(pred_dict[i][7])
                per_obj.id = int(pred_dict[i][7])
                label = int(pred_dict[i][9])
            else:
                marker.id = i
                per_obj.id = i
                label = int(pred_dict[i][8])
            marker.color.a = 0.6

            marker.color.r = class_colors[label][0]
            marker.color.g = class_colors[label][1]
            marker.color.b = class_colors[label][2]
            marker.scale.x = pred_dict[i][4]
            marker.scale.y = pred_dict[i][5]
            marker.scale.z = pred_dict[i][6]
            quaternion = euler_to_quaternion(pred_dict[i][3].item(),0, 0)
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]
            marker.pose.position.x = pred_dict[i][0]
            marker.pose.position.y = pred_dict[i][1]
            marker.pose.position.z = pred_dict[i][2]

            marker.header.stamp=timestamp
            marker.lifetime = rospy.Duration(0,100000000)
            markerArray.markers.append(marker)

            per_obj.bounding_box.x = pred_dict[i][4]
            per_obj.bounding_box.y = pred_dict[i][5]
            per_obj.bounding_box.z = pred_dict[i][6]
            per_obj.position.x = pred_dict[i][0]
            per_obj.position.y = pred_dict[i][1]
            per_obj.position.z = pred_dict[i][2]
            per_obj.velocity.x = 0.0
            per_obj.velocity.y = 0.0
            per_obj.velocity.z = 0.0
            per_obj.heading = pred_dict[i][3]
            per_obj.type = label
            percep_objs.objects.append(per_obj)

        self.publisher_marks.publish(markerArray)
        self.publisher_objs.publish(percep_objs)