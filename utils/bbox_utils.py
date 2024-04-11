def get_bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    center_x = int((x1+x2)/2)
    center_y = int((y1+y2)/2)
    return center_x, center_y

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 #pytagoras theorem for hypothenus

def get_foot_pos(bbox):
    x1,y1,x2,y2 = bbox
    return(int((x1+x2)/2), y2)

def get_closest_keypoint_idx(point, kps, kps_idx):
    closest_distance = float('inf')
    kps_index = kps_idx[0]
    for keypoint_index in kps_idx:
        keypoint = kps[keypoint_index*2], kps[keypoint_index*2+1]
        distance = abs(point[1]-keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            kps_index = keypoint_index

    return kps_index

def get_bbox_height(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_bbox_center(bbox):
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))