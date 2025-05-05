import cv2
import numpy as np
from ultralytics import YOLO

# --- Параметры ---
FONT_SCALE = 1.5
TEXT_THICKNESS = 3
MARKER_SIZE = 10
LINE_THICKNESS = 2
MAX_DISPLAY_HEIGHT = 1080
SQUARE_SIZE = 2.9  # см, размер клетки шахматной доски (проверь реальный размер!)
YOLO_CUP_CLASS_ID = 41  # класс "cup" в YOLOv8

# --- Загрузка изображения ---
image = cv2.imread('./fotn/1.jpg')
orig_height, orig_width = image.shape[:2]

# --- Параметры камеры ---
camera_params = {
    'fx': 3185.51521,
    'fy': 3183.20384,
    'cx': 1970.84117,
    'cy': 1478.60279,
    'dist_coeffs': np.array([0.119781817, -0.883606827, -0.00250506375, 0.000368101997, 1.84805235])
}

# --- Матрица камеры ---
camera_matrix = np.array([
    [camera_params['fx'], 0, camera_params['cx']],
    [0, camera_params['fy'], camera_params['cy']],
    [0, 0, 1]
], dtype=np.float32)

# --- Цвета ---
COLORS = {
    'board1': (255, 0, 255),
    'board2': (0, 255, 255),
    'object': (212, 255, 127),
    'text':   (255, 255, 255)
}

# --- Функции ---

def undistort_points(points):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, camera_matrix, camera_params['dist_coeffs'], P=camera_matrix).reshape(-1, 2)

def find_chessboard(gray, chessboard_size=(7, 6), mask=None):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * SQUARE_SIZE

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Здесь исправленный вызов — mask передаётся позиционно
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, mask, flags)

    if not found:
        return None

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    undistorted_corners = undistort_points(corners)

    ret, rvec, tvec = cv2.solvePnP(objp, undistorted_corners, camera_matrix, camera_params['dist_coeffs'])
    if not ret:
        print("solvePnP failed")
        return None

    Rmat = cv2.Rodrigues(rvec)[0]
    print(f"Board tvec: {tvec.flatten()}")
    return {
        'found': True,
        'objp': objp,
        'corners': corners,
        'undistorted_corners': undistorted_corners,
        'rvec': rvec,
        'tvec': tvec,
        'Rmat': Rmat
    }

def project_board_center(data):
    board_center_3d = np.array([[SQUARE_SIZE * (data['objp'][-1][0] / SQUARE_SIZE) / 2,
                                 SQUARE_SIZE * (data['objp'][-1][1] / SQUARE_SIZE) / 2,
                                 0]], dtype=np.float32)
    projected_center, _ = cv2.projectPoints(board_center_3d, data['rvec'], data['tvec'], camera_matrix, camera_params['dist_coeffs'])
    return tuple(projected_center.reshape(-1, 2).astype(int)[0])

# --- Обнаружение досок ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
board1_data = find_chessboard(gray)

mask = np.ones_like(gray) * 255
if board1_data and board1_data['found']:
    cv2.fillPoly(mask, [np.int32(board1_data['corners'])], 0)

gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
board2_data = find_chessboard(gray_masked)

boards = []
for idx, data in enumerate([board1_data, board2_data]):
    if not data or not data.get('found'):
        continue
    center = project_board_center(data)
    distance = np.linalg.norm(data['tvec'])
    color = COLORS['board1'] if idx == 0 else COLORS['board2']
    boards.append({'center': center, 'distance': distance, 'data': data})
    cv2.circle(image, center, MARKER_SIZE, color, -1)
    cv2.putText(image, f"Board {idx+1}: {distance*100:.1f} cm", (center[0]-150, center[1]+(75 if idx==0 else -75)),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, TEXT_THICKNESS)

# --- Обнаружение кружки ---
model = YOLO("yolov8l.pt")
results = model(image)
O = None

for result in results:
    for box in result.boxes:
        if int(box.cls[0]) == YOLO_CUP_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            O = ((x1 + x2) // 2, (y1 + y2) // 2)
            undistorted_obj_center = undistort_points([O])[0]
            O = (int(undistorted_obj_center[0]), int(undistorted_obj_center[1]))
            cv2.rectangle(image, (x1, y1), (x2, y2), COLORS['object'], LINE_THICKNESS)
            cv2.circle(image, O, MARKER_SIZE, COLORS['object'], -1)
            cv2.putText(image, f"Cup", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['object'], TEXT_THICKNESS)
            cv2.putText(image, f"({O[0]}, {O[1]})", (O[0], O[1] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['object'], TEXT_THICKNESS)

# --- Триангуляция кружки по двум доскам ---
if len(boards) >= 2 and O is not None:
    b1 = boards[0]['data']
    b2 = boards[1]['data']

    # Проекционные матрицы
    P1 = camera_matrix @ np.hstack((b1['Rmat'], b1['tvec']))
    P2 = camera_matrix @ np.hstack((b2['Rmat'], b2['tvec']))

    point1 = np.array([[O[0], O[1]]], dtype=np.float32)
    point2 = point1.copy()

    # Преобразование точек в формат OpenCV для триангуляции
    point1_input = point1.reshape(-1, 1, 2)
    point2_input = point2.reshape(-1, 1, 2)

    points_4d = cv2.triangulatePoints(P1, P2, point1_input, point2_input)
    points_3d = points_4d[:3] / points_4d[3]

    distance_to_cup = np.linalg.norm(points_3d)

    print(f"3D координаты кружки: {points_3d.flatten()}")
    print(f"Расстояние до кружки: {distance_to_cup:.2f} м")

    text_offset_x, text_offset_y = 50, 950
    cv2.putText(image, f"Triangulated Cup Distance: {distance_to_cup*100:.1f} cm",
                (text_offset_x, text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)

# --- Отображение результата ---
scale_factor = min(MAX_DISPLAY_HEIGHT / orig_height, 1.0)
display_width = int(orig_width * scale_factor)
display_height = int(orig_height * scale_factor)
resized_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.imshow('Result', resized_image)
cv2.resizeWindow('Result', display_width, display_height)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Сохранение результата ---
cv2.imwrite('full_res_result.jpg', image)
cv2.imwrite('display_result.jpg', resized_image)