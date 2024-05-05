import numpy as np 
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
    return tuple(np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.int32))

def get_bbox_width(bbox):
    return (bbox[2]-bbox[0])
