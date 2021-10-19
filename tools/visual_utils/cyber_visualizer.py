from perception_object_pb2 import PerceptionObjects,PerceptionObject


class Visualizer(object):
    def __init__(self,cyber_node):
        self.publisher_objs = cyber_node.create_writer("/perception/objects",PerceptionObjects,1)

    def pub(self,pred_dict,stamp):
        obj_size = pred_dict.shape[0]
        is_tracked = pred_dict.shape[1] == 10
        percep_objs = PerceptionObjects()
        percep_objs.time_meas = stamp

        for i in range(obj_size):
            per_obj = PerceptionObject()
            if is_tracked:
                per_obj.id = int(pred_dict[i][7])
                label = int(pred_dict[i][9])
            else:
                per_obj.id = i
                label = int(pred_dict[i][8])


            per_obj.bounding_box.x = pred_dict[i][4]
            per_obj.bounding_box.y = pred_dict[i][5]
            per_obj.bounding_box.z = pred_dict[i][6]
            per_obj.position.x = pred_dict[i][0]
            per_obj.position.y = pred_dict[i][1]
            per_obj.position.z = pred_dict[i][2]
            # per_obj.velocity.x = 0.0
            # per_obj.velocity.y = 0.0
            # per_obj.velocity.z = 0.0
            per_obj.heading = pred_dict[i][3]
            per_obj.type = label
            percep_objs.objects.append(per_obj)

        self.publisher_objs.write(percep_objs)