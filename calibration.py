import cv2
import numpy as np
import glob

# Размер шахматной доски
chessboard_size = (7, 6)
square_size = 1.0  # размер клетки в см или метрах

# Подготовка списков для 3D и 2D точек
objpoints = []  # 3D точки в реальном мире
imgpoints = []  # 2D точки на изображении

# Создание объектных точек (Z=0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Загрузка изображений
images = glob.glob('./fotn/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск углов шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Калибровка камеры
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Матрица камеры:\n", camera_matrix)
print("Коэффициенты искажения:\n", dist_coeffs)