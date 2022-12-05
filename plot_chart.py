import matplotlib.pyplot as plt
SCRFD_x = [3.6, 4.2]
SCRFD_y = [68.15, 77.87]
RetinaFace_x = [7.9]
RetinaFace_y = [47.32]
YOLOv5nFace_x = [3.7, 4.8]
YOLOv5nFace_y = [73.82, 80.53]
Our_YOLOv7Face_x = [5.3]
Our_YOLOv7Face_y = [81.45]

plt.plot(SCRFD_x, SCRFD_y, marker='s', markerfacecolor='blue', markersize=10, linestyle='dashed', color='blue', label="SCRFD")
plt.plot(RetinaFace_x, RetinaFace_y, marker='^', markerfacecolor='cyan', markersize=10, linestyle='dashed', color='cyan', label="RetinaFace")
plt.plot(YOLOv5nFace_x, YOLOv5nFace_y, marker='o', markerfacecolor='green', markersize=10, linestyle='dashed', color="green", label="YOLOv5Face")
plt.plot(Our_YOLOv7Face_x, Our_YOLOv7Face_y, marker='D', markerfacecolor='red', markersize=10, linestyle='dashed', color="red", label="VILOFACE")

plt.xlim(left=3, right=10)
plt.ylim(bottom=30, top=100)
plt.xlabel('<-    Lower is better         Inference time(ms)                          ')
plt.ylabel('AP on WIDERFACE Hard Val (%)    Higher is better     ->')
plt.grid()
plt.legend()
plt.show()