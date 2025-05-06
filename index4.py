import cv2
import numpy as np
from ultralytics import YOLO

# --- Параметры ---
SQUARE_SIZE = 2.9  # см, размер клетки шахматной доски
YOLO_CUP_CLASS_ID = 41  # класс "cup" в YOLOv8

# --- Параметры камеры ---
camera_params = {
    'fx': 2881.55270,
    'fy': 2903.39102,
    'cx': 1578.30063,
    'cy': 1546.68121,
    'dist_coeffs': np.array([-0.34934032, 0.27588199, -0.00491737, 0.02455157, -0.13104934])
}

# --- Матрица камеры ---
camera_matrix = np.array([
    [camera_params['fx'], 0, camera_params['cx']],
    [0, camera_params['fy'], camera_params['cy']],
    [0, 0, 1]
], dtype=np.float32)

# --- Функции ---

def undistort_points(points):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(points, camera_matrix, camera_params['dist_coeffs'], P=camera_matrix).reshape(-1, 2)

def find_chessboard(gray, chessboard_size=(7, 6), mask=None):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * SQUARE_SIZE

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
    return {
        'found': True,
        'objp': objp,
        'corners': corners,
        'undistorted_corners': undistorted_corners,
        'rvec': rvec,
        'tvec': tvec,
        'Rmat': Rmat
    }

def main():
    image = cv2.imread('./fotn/1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Улучшение контрастности
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 5)  # Сглаживание шумов

    # Поиск первой доски
    board1_data = find_chessboard(gray)
    if board1_data and board1_data['found']:
        print("[+] Первая доска найдена.")
        cv2.drawChessboardCorners(image, (7, 6), board1_data['corners'], True)
        cv2.imshow('Board 1', image)
        cv2.waitKey(0)

        # Создание маски для второй доски
        mask = np.ones_like(gray) * 255
        cv2.fillPoly(mask, [np.int32(board1_data['corners'])], 0)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

        # Показываем изображение после маскирования
        cv2.imshow('Masked Image', gray_masked)
        cv2.waitKey(0)

        # Поиск второй доски
        board2_data = find_chessboard(gray_masked)
        if board2_data and board2_data['found']:
            print("[+] Вторая доска найдена.")
            cv2.drawChessboardCorners(image, (7, 6), board2_data['corners'], True)
            cv2.imshow('Board 2', image)
            cv2.waitKey(0)

    else:
        print("[-] Первая доска не найдена.")

    # Обнаружение кружки
    model = YOLO("yolov8l.pt")
    results = model(image)
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == YOLO_CUP_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "Cup", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Detected Cup', image)
                cv2.waitKey(0)

if __name__ == "__main__":
    main()