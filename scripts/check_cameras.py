"""
检查所有可用摄像头的 index 和参数
运行: python check_cameras.py
"""

import cv2
import os

SAVE_DIR = os.path.expanduser("~/camera_check")
os.makedirs(SAVE_DIR, exist_ok=True)

RESOLUTIONS = [
    (1920, 1080),
    (1280, 720),
    (640, 480),
    (640, 360),
]

found = []

print("正在扫描摄像头 (index 0~10)...\n")

for idx in range(11):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        continue

    # 尝试读一帧确认真的能用
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        continue

    # 获取默认分辨率
    default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"✅ 找到相机 index={idx}")
    print(f"   默认分辨率: {default_w}x{default_h} @ {default_fps:.0f}fps")

    # 保存默认分辨率下的图片
    img_path = os.path.join(SAVE_DIR, f"camera_{idx}_default_{default_w}x{default_h}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"   图片已保存: {img_path}")

    # 测试支持哪些分辨率
    supported = []
    for w, h in RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            supported.append(f"{w}x{h}")

    print(f"   支持的分辨率: {', '.join(supported) if supported else '仅默认'}")

    found.append(idx)
    cap.release()
    print()

if not found:
    print("❌ 未找到任何摄像头，请检查相机权限（系统设定 → 隐私权与安全性 → 相机 → 终端机）")
else:
    print(f"共找到 {len(found)} 个摄像头，index: {found}")
    print(f"预览图片保存在: {SAVE_DIR}")
    print("\n用法示例（lerobot-record）:")
    for i, idx in enumerate(found):
        name = ["handeye", "fixed", "extra"][i] if i < 3 else f"cam{i}"
        print(f"  {name}: index_or_path={idx}")

    # 实时预览所有相机
    print("\n开启实时预览，按 Q 退出...")
    caps = {idx: cv2.VideoCapture(idx) for idx in found}

    while True:
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {idx}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
