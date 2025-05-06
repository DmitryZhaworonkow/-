import cv2
import numpy as np
from ultralytics import YOLO

# --- Параметры камеры ---
camera_params = {
    'fx': 2881.55270,
    'fy': 2903.39102,
    'cx': 1578.30063,
    'cy': 1546.68121,
    'dist_coeffs': np.array([-0.34934032, 0.27588199, -0.00491737, 0.02455157, -0.13104934])
}

camera_matrix = np.array([
    [camera_params['fx'], 0, camera_params['cx']],
    [0, camera_params['fy'], camera_params['cy']],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = camera_params['dist_coeffs']

# --- Размер шахматной доски ---
pattern_size = (7, 6)
square_size_cm = 2.9  # размер клетки в см

# --- Реальные координаты точек доски ---
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * square_size_cm

# --- Загрузка модели YOLO ---
model = YOLO("yolov8l.pt")

# --- Путь к изображению ---
image_path = "./fotn/1.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("❌ Ошибка: не удалось загрузить изображение")
    exit()

# --- Поиск двух досок ---
def find_two_chessboards(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_list = []
    frame_masked = image.copy()

    for _ in range(2):
        ret, corners = cv2.findChessboardCorners(cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY), pattern_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            mask = np.zeros_like(gray)
            points = corners.reshape(-1, 2).astype(np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], (255, 255, 255))
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
            frame_masked = cv2.bitwise_and(frame_masked, frame_masked, mask=mask)
            corners_list.append(corners)
        else:
            print("❌ Доска не найдена")
            break

    return corners_list

corners_list = find_two_chessboards(frame)

# --- Решение PnP для обеих досок ---
rvecs = []
tvecs = []

for i, corners in enumerate(corners_list):
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if success:
        rvecs.append(rvec)
        tvecs.append(tvec)
        print(f"✅ Доска {i+1}: Расстояние до камеры — {np.linalg.norm(tvec):.2f} см")
    else:
        print(f"❌ Не удалось найти позицию доски {i+1}")

# --- Обнаружение кружки ---
results = model(frame)
cup_center = None

for result in results:
    boxes = result.boxes
    for box in boxes:
        if box.cls == 41:  # класс "cup"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cup_bottom_point = ((x1 + x2) // 2, y2)

# --- Функция вычисления плоскости пола по двум доскам ---
def get_floor_plane_equation(objp, rvecs, tvecs):
    all_corners_3d = []

    for i in range(len(rvecs)):
        corners_3d, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        corners_3d = corners_3d.reshape(-1, 3)
        all_corners_3d.append(corners_3d)

    all_corners_3d = np.vstack(all_corners_3d)
    p1, p2, p3 = all_corners_3d[:3]

    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    d = -np.dot(normal, p1)
    return normal, d  # ax + by + cz + d = 0

# --- Перевод точки кружки на плоскость пола ---
def point_on_floor_plane(point, rvec, tvec, floor_normal, floor_d, camera_matrix, dist_coeffs):
    point_undistorted = cv2.undistortPoints(np.array([point], dtype=np.float32), camera_matrix, dist_coeffs, P=camera_matrix)

    rot_mat, _ = cv2.Rodrigues(rvec)
    inv_rot = rot_mat.T
    inv_cam = np.linalg.inv(camera_matrix)

    dir_vector = inv_rot @ inv_cam @ np.array([*point_undistorted[0, 0], 1]).reshape(3, 1)
    origin = tvec.flatten()

    denom = np.dot(floor_normal, dir_vector)
    if abs(denom) < 1e-6:
        return None

    t = -(np.dot(floor_normal, origin) + floor_d) / denom
    cup_position = origin + t * dir_vector.flatten()
    return cup_position

# --- Вычисление угла между направлением камеры и полом ---
def get_camera_angle_to_floor(rvec, floor_normal):
    # Направление взгляда камеры (Z=1)
    cam_dir = np.array([0, 0, 1], dtype=np.float32)

    # Вращаем направление камеры в мировую систему координат
    rot_mat, _ = cv2.Rodrigues(rvec)
    cam_dir_world = rot_mat.T @ cam_dir  # камера -> мир

    # Угол между лучом камеры и нормалью пола
    cos_angle = np.dot(cam_dir_world, floor_normal) / (np.linalg.norm(cam_dir_world) * np.linalg.norm(floor_normal))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg