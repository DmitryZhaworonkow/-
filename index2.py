import cv2
import numpy as np
from ultralytics import YOLO
import math
import argparse

# --- Параметры ---
FONT_SCALE = 1.0
TEXT_THICKNESS = 2
LINE_THICKNESS = 2
MARKER_SIZE = 6
MAX_DISPLAY_HEIGHT = 720
SQUARE_SIZE = 2.9  # см

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


def draw_text(img, text, pos, color=COLORS['text'], scale=FONT_SCALE, thickness=TEXT_THICKNESS):
    """Универсальная функция для отрисовки текста"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def find_chessboard(gray, chessboard_size=(7, 6), mask=None):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * SQUARE_SIZE

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    found, corners = cv2.findChessboardCorners(gray, chessboard_size, mask, flags)
    if not found:
        return None

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    undistorted_corners = cv2.undistortPoints(corners, camera_matrix, camera_params['dist_coeffs'], P=camera_matrix)

    ret, rvec, tvec = cv2.solvePnP(objp, undistorted_corners, camera_matrix, camera_params['dist_coeffs'])
    if not ret:
        return None

    Rmat, _ = cv2.Rodrigues(rvec)
    world_points = (Rmat @ objp.T).T + tvec.flatten()
    plane_point = np.mean(world_points, axis=0)

    return {
        'found': True,
        'objp': objp,
        'corners': corners,
        'rvec': rvec,
        'tvec': tvec,
        'Rmat': Rmat,
        'world_points': world_points,
        'plane_point': plane_point
    }


def compute_plane_normal(data):
    centered = data['world_points'] - data['plane_point']
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    normal /= np.linalg.norm(normal)

    normal[1], normal[2] = normal[2], normal[1]
    normal[1] *= -1
    return normal


def align_camera_with_floor(boards):
    normals = [b['normal'] for b in boards]
    if len(normals) < 2:
        return boards

    floor_normal = np.cross(normals[0], normals[1])
    floor_normal /= np.linalg.norm(floor_normal)

    up_vector = np.array([0, 1, 0], dtype=np.float32)
    angle_deg = math.degrees(np.arccos(np.clip(np.dot(floor_normal, up_vector), -1.0, 1.0)))

    print(f"\nНормаль пола: {floor_normal}")
    print(f"Угол между полом и осью Y: {angle_deg:.1f}°")

    if abs(angle_deg) > 5:
        axis = np.cross(floor_normal, up_vector)
        axis /= np.linalg.norm(axis)
        correction_R = cv2.Rodrigues(math.radians(angle_deg) * axis)[0]

        for board in boards:
            board['data']['Rmat'] = correction_R @ board['data']['Rmat']
            board['data']['tvec'] = (correction_R @ board['data']['tvec'].flatten()).reshape(3, 1)

    return boards


def calculate_object_distance(board, O):
    x_norm = (O[0] - camera_params['cx']) / camera_params['fx']
    y_norm = (O[1] - camera_params['cy']) / camera_params['fy']
    direction = np.array([x_norm, y_norm, 1], dtype=np.float32)
    direction /= np.linalg.norm(direction)

    data = board['data']
    normal = board['normal']
    plane_point = data['plane_point']

    denominator = np.dot(normal, direction)
    if abs(denominator) < 1e-6:
        return None, None

    t = np.dot(normal, plane_point) / denominator
    point = direction * t
    distance = np.linalg.norm(point)
    return distance, point


def triangulate_position(board1, board2, O):
    dist1, pt1 = calculate_object_distance(board1, O)
    dist2, pt2 = calculate_object_distance(board2, O)

    if pt1 is None or pt2 is None:
        return pt1 if pt1 is not None else pt2

    return (pt1 + pt2) / 2


def main(image_path):
    image = cv2.imread(f'fotn/{image_path}.jpg')
    orig_height, orig_width = image.shape[:2]

    undistorted_image = cv2.undistort(image, camera_matrix, camera_params['dist_coeffs'])
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

    board1_data = find_chessboard(gray)
    mask = np.ones_like(gray) * 255

    if board1_data and board1_data['found']:
        cv2.fillPoly(mask, [np.int32(board1_data['corners'])], 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=1)

    gray_masked = cv2.bitwise_and(src1=gray, src2=gray, mask=mask)
    board2_data = find_chessboard(gray_masked)

    boards = []
    for idx, data in enumerate([board1_data, board2_data]):
        if not data or not data.get('found'):
            continue

        center = tuple(np.int32(np.mean(data['corners'], axis=0).ravel()))
        normal = compute_plane_normal(data)
        boards.append({
            'center': center,
            'data': data,
            'normal': normal
        })

    if len(boards) < 1:
        print("Не найдено ни одной доски.")
        return

    boards = align_camera_with_floor(boards)

    print("\n--- Координаты досок в скорректированной системе ---")
    for idx, board in enumerate(boards):
        data = board['data']
        world_pos = data['tvec'].flatten()
        distance = np.linalg.norm(world_pos)

        print(f"Доска {idx+1}:")
        print(f"  Положение в пространстве: X={world_pos[0]:.2f}, Y={world_pos[1]:.2f}, Z={world_pos[2]:.2f} см")
        print(f"  Расстояние от камеры: {distance:.2f} см")

    model = YOLO("yolov8l.pt")
    results = model(undistorted_image)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            O = ((x1 + x2) // 2, (y1 + y2) // 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), COLORS['object'], LINE_THICKNESS)
            cv2.circle(image, O, MARKER_SIZE, COLORS['object'], -1)
            draw_text(image, f"{label}", (x1, y1 - 30), COLORS['object'], FONT_SCALE, TEXT_THICKNESS)
            draw_text(image, f"({O[0]}, {O[1]})", (O[0], O[1] - 80), COLORS['object'])

            if len(boards) >= 2:
                point_3d = triangulate_position(boards[0], boards[1], O)
                if point_3d is not None:
                    distance = np.linalg.norm(point_3d)
                    print(f"\nОбъект '{label}':")
                    print(f"  Расстояние: {distance:.2f} см")
                    print(f"  Координаты: X={point_3d[0]:.2f}, Y={point_3d[1]:.2f}, Z={point_3d[2]:.2f}")

                    draw_text(image, f"Distance: {distance:.1f} cm", (x1, y1 - 120), COLORS['text'])
                    draw_text(image, f"3D: ({point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f})", (x1, y1 - 80), COLORS['text'])

    for idx, board in enumerate(boards):
        color = COLORS['board1'] if idx == 0 else COLORS['board2']
        center = board['center']
        data = board['data']
        rvec, tvec = data['rvec'], data['tvec']
        cv2.circle(image, center, MARKER_SIZE, color, -1)
        draw_text(image, f"Board {idx+1}", (center[0]-150, center[1]+(75 if idx == 0 else -75)), color)
        cv2.drawFrameAxes(image, camera_matrix, camera_params['dist_coeffs'], rvec, tvec, length=5)

    scale_factor = min(MAX_DISPLAY_HEIGHT / orig_height, 1.0)
    display_size = (int(orig_width * scale_factor), int(orig_height * scale_factor))
    resized_image = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('full_res_result.jpg', image)
    cv2.imwrite('display_result.jpg', resized_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Определение позиции объектов на основе шахматных досок")
    parser.add_argument("--image", type=int, default=1, help="Путь к изображению")
    args = parser.parse_args()

    main(args.image)