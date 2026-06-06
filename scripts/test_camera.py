from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
import cv2

config = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=640,
    height=480,
    fourcc="MJPG",
    color_mode="bgr"
)

camera = OpenCVCamera(config)
camera.connect()

print("相机已连接，按 q 退出")

while True:
    frame = camera.read()
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.disconnect()
cv2.destroyAllWindows()
