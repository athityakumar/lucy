import numpy as np

r_y1 = 10
r_y2 = 100
rois_per_box = 5
threshold = 20
count = 10

rois = np.zeros((count, 2), dtype=np.int32)

y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
print(y1y2)

y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
print(y1y2)

y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
print(y1)
print(y2)

box_rois = np.hstack([y1, y2])
print(box_rois)

i = 1
try:
    rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois
    print(rois)
except:
    print("Some issue")