import cv2
import numpy as np
from ultralytics import YOLO
import math
import threading
import queue

# --- Константы ---
FONT_SCALE = 1.5
TEXT_THICKNESS = 3
LINE_THICKNESS = 2
MARKER_SIZE = 10
MAX_DISPLAY_HEIGHT = 720
SQUARE_SIZE = 2.9  # см
CACHE_INTERVAL = 10  # обновлять данные о досках каждые N кадров
DOWNSCALE_FACTOR = 0.5  # уменьшаем изображение для ускорения findChessboardCorners

# --- Цвета ---
COLORS = {
    'board1': (255, 0, 255),
    'board2': (0, 255, 255),
    'object': (212, 255, 127),
    'text': (255, 255, 255)
}

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

# --- Загрузка модели YOLO один раз ---
model = YOLO("yolov8l.pt")

# --- Очереди для передачи данных между потоками ---
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# --- Функции ---

def undistort_points(points):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, camera_matrix, camera_params['dist_coeffs'], P=camera_matrix).reshape(-1, 2)

def find_chessboard_cached(gray, chessboard_size=(7, 6), downscale=DOWNSCALE_FACTOR):
    h, w = gray.shape
    small_gray = cv2.resize(gray, (int(w * downscale), int(h * downscale)), interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small_gray, chessboard_size, None, flags)
    if not found:
        return None

    corners = corners / downscale  # восстановить масштаб
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * SQUARE_SIZE
    objp = np.hstack((objp, np.zeros((objp.shape[0], 1))))

    undistorted_corners = undistort_points(corners)
    ret, rvec, tvec = cv2.solvePnP(objp, undistorted_corners, camera_matrix, camera_params['dist_coeffs'])
    if not ret:
        return None

    Rmat = cv2.Rodrigues(rvec)[0]

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
    board_center_3d = np.array([[data['objp'][-1][0] / 2, data['objp'][-1][1] / 2, 0]], dtype=np.float32)
    projected_center, _ = cv2.projectPoints(board_center_3d, data['rvec'], data['tvec'], camera_matrix, camera_params['dist_coeffs'])
    return tuple(projected_center.reshape(-1, 2).astype(int)[0])

def draw_board_info(image, center, distance, idx, color):
    offset_y = 75 if idx == 0 else -75
    cv2.circle(image, center, MARKER_SIZE, color, -1)
    cv2.putText(image, f"Board {idx+1}: {distance*100:.1f} cm",
                (center[0]-150, center[1]+offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, TEXT_THICKNESS)

# --- Поток для детекции объектов ---
def yolo_thread():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame)
            result_queue.put((frame.copy(), results))

# --- Запуск потока YOLO ---
threading.Thread(target=yolo_thread, daemon=True).start()

# --- Открытие видеопотока ---
cap = cv2.VideoCapture(0)

frame_counter = 0
board1_data_cached = None
board2_data_cached = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Передача кадра в поток YOLO ---
    if frame_queue.full():
        frame_queue.get()
    frame_queue.put(frame.copy())

    # --- Получение результатов YOLO ---
    if not result_queue.empty():
        processed_frame, results = result_queue.get()

        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        boards = []
        mask = np.ones_like(gray) * 255

        # --- Поиск досок только с периодичностью ---
        if frame_counter % CACHE_INTERVAL == 0:
            board1_data_cached = find_chessboard_cached(gray)
            if board1_data_cached:
                cv2.fillPoly(mask, [np.int32(board1_data_cached['corners'])], 0)
                gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
                board2_data_cached = find_chessboard_cached(gray_masked)

        # --- Обработка найденных досок ---
        for idx, data in enumerate([board1_data_cached, board2_data_cached]):
            if not data or not data.get('found'):
                continue
            center = project_board_center(data)
            distance = np.linalg.norm(data['tvec'])
            color = COLORS[f'board{idx+1}']
            boards.append({'center': center, 'distance': distance, 'data': data})
            draw_board_info(processed_frame, center, distance, idx, color)

        # --- Обнаружение кружки ---
        O = None
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 41:  # класс "cup"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    O = ((x1 + x2) // 2, (y1 + y2) // 2)
                    undistorted_obj_center = undistort_points([O])[0]
                    O = tuple(map(int, undistorted_obj_center))

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), COLORS['object'], LINE_THICKNESS)
                    cv2.circle(processed_frame, O, MARKER_SIZE, COLORS['object'], -1)
                    cv2.putText(processed_frame, "Cup", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['object'], TEXT_THICKNESS)
                    cv2.putText(processed_frame, f"({O[0]}, {O[1]})", (O[0], O[1] - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['object'], TEXT_THICKNESS)

        # --- Расчёт расстояния до кружки ---
        if boards and O is not None:
            b1 = boards[0]['data']
            tvec1 = b1['tvec']

            K_x, K_y = camera_params['cx'], camera_params['cy']
            O_x, O_y = O

            angle_x = math.atan((O_x - K_x) / camera_params['fx'])
            angle_y = math.atan((O_y - K_y) / camera_params['fy'])

            z_distance = abs(tvec1[2][0])
            distance_to_cup = z_distance / math.cos(math.sqrt(angle_x**2 + angle_y**2))

            cv2.putText(processed_frame, f"Estimated Cup Distance: {distance_to_cup:.1f} cm",
                        (50, 950), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)

        # --- Отображение результата ---
        orig_height, orig_width = processed_frame.shape[:2]
        scale_factor = min(MAX_DISPLAY_HEIGHT / orig_height, 1.0)
        display_dim = (int(orig_width * scale_factor), int(orig_height * scale_factor))
        resized_image = cv2.resize(processed_frame, display_dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('Optimized Video Stream', resized_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()