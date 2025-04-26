import cv2
import numpy as np
from ultralytics import YOLO
import math 

# Параметры отрисовки
FONT_SCALE = 2
TEXT_THICKNESS = 4
LINE_THICKNESS = 2
MARKER_SIZE = 10
MAX_DISPLAY_HEIGHT = 1080

# Загрузка изображения
image = cv2.imread('./fotn/1.jpg')
orig_height, orig_width = image.shape[:2]

# Параметры камеры
camera_params = {
    'fx': 3.18551521e+03,
    'fy': 3.18320384e+03,
    'cx': 1.97084117e+03,
    'cy': 1.47860279e+03,
    'dist_coeffs': np.array([1.19781817e-01, -8.83606827e-01, -2.50506375e-03, 3.68101997e-04, 1.84805235e+00])
}

# Матрица камеры
camera_matrix = np.array([
    [camera_params['fx'], 0, camera_params['cx']],
    [0, camera_params['fy'], camera_params['cy']],
    [0, 0, 1]
], dtype=np.float32)

def undistort_points(points):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(
        points, 
        camera_matrix, 
        camera_params['dist_coeffs'],
        P=camera_matrix
    )
    return undistorted_points.reshape(-1, 2)

def calculate_distance_tvec(tvec):
    # Расчет расстояния через tvec
    tx, ty, tz = tvec.flatten()
    distance = np.sqrt(tx**2 + ty**2 + tz**2)
    return distance

# Цвета для отрисовки
COLORS = {
    'board1': (255, 0, 255),
    'board2': (0, 255, 255),
    'object': (212, 255, 127),
    'text':   (255, 255, 255)
}

def calculate_distance(corners, chessboard_size, square_size, camera_params):
    h_pix = abs(corners[0][1] - corners[-1][1])  # Используем исправленные координаты
    fx = camera_params['fx']
    cx = camera_params['cx']
    psi = np.arctan(cx / fx) * 2
    R = gray.shape[0]
    h = (chessboard_size[1] + 1) * square_size
    d = (R / (2 * np.tan(psi / 2))) * (h / h_pix)
    center = np.mean(corners, axis=0)
    return d, (int(center[0]), int(center[1]))  # Расстояние и координаты центра
# Функции для расчета углов

def calculate_angle_s1(D1, O, K):
    v1 = np.array(O) - np.array(D1)
    v2 = np.array(K) - np.array(D1)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_s1 = dot_product / (norm_v1 * norm_v2)
    s1 = np.arccos(np.clip(cos_s1, -1.0, 1.0))
    return np.degrees(s1), norm_v1, norm_v2, cos_s1

def calculate_angle_s2(D2, O, K):
    v1 = np.array(O) - np.array(D2)
    v2 = np.array(K) - np.array(D2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_s2 = dot_product / (norm_v1 * norm_v2)
    s2 = np.arccos(np.clip(cos_s2, -1.0, 1.0))
    return np.degrees(s2), norm_v1, norm_v2, cos_s2

def calculate_angle_s3(D1, O, K):
    v1 = np.array(D1) - np.array(K)
    v2 = np.array(O) - np.array(K)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_s3 = dot_product / (norm_v1 * norm_v2)
    s3 = np.arccos(np.clip(cos_s3, -1.0, 1.0))
    return np.degrees(s3), norm_v1, norm_v2, cos_s3

def calculate_angle_s4(D2, O, K):
    v1 = np.array(O) - np.array(K)
    v2 = np.array(D2) - np.array(K)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_s4 = dot_product / (norm_v1 * norm_v2)
    s4 = np.arccos(np.clip(cos_s4, -1.0, 1.0))
    return np.degrees(s4),norm_v1, norm_v2, cos_s4

def calculate_angle_s5(D1, O, K):
    v1 = np.array(D1) - np.array(O)
    v2 = np.array(K) - np.array(O)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_s5 = dot_product / (norm_v1 * norm_v2)
    s5 = np.arccos(np.clip(cos_s5, -1.0, 1.0))
    return np.degrees(s5), norm_v1, norm_v2, cos_s5

def calculate_angle_s6(D2, O, K):
    v1 = np.array(K) - np.array(O)
    v2 = np.array(D2) - np.array(O)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_s6 = dot_product / (norm_v1 * norm_v2)
    s6 = np.arccos(np.clip(cos_s6, -1.0, 1.0))
    return np.degrees(s6), norm_v1, norm_v2, cos_s6

def calculate_angle_s7(D11, K11, D22):
    v1 = np.array(D11) - np.array(K11)
    v2 = np.array(D22) - np.array(K11)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_s7 = dot_product / (norm_v1 * norm_v2)
    s7 = np.arccos(np.clip(cos_s7, -1.0, 1.0))
    return np.degrees(s7)

# Поиск шахматных досок
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
chessboard_size = (7, 6)
square_size = 2.9  # см

# Создаем 3D-точки для шахматной доски
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Найти первую шахматную доску
found1, corners1 = cv2.findChessboardCorners(gray, chessboard_size, flags)
rvec1 = tvec1 = None
if found1:
    corners1 = cv2.cornerSubPix(gray, corners1, (11, 11), (-1, -1), criteria)
    undistorted_corners1 = undistort_points(corners1)
    
    
    # Рассчитываем 3D-позицию первой доски
    ret1, rvec1, tvec1 = cv2.solvePnP(objp, undistorted_corners1, camera_matrix, camera_params['dist_coeffs'])

    # Матрица вращения для первой доски
    Rvec1=cv2.Rodrigues(rvec1)[0]
    #print(Rvec1)
    
    # Расчет расстояния через tvec
    distance1 = calculate_distance_tvec(tvec1)
    dist1, center1 = calculate_distance(undistorted_corners1, chessboard_size, square_size, camera_params)
    
    # Проектируем 3D-центр доски обратно в 2D
    board_center_3d = np.array([[(chessboard_size[0] - 1) * square_size / 2, 
                                 (chessboard_size[1] - 1) * square_size / 2, 
                                 0]], dtype=np.float32)
    projected_center1, _ = cv2.projectPoints(board_center_3d, rvec1, tvec1, camera_matrix, camera_params['dist_coeffs'])
    D1 = projected_center1.reshape(-1, 2).astype(int)[0]
    
    # Отрисовка 2D-центра
    cv2.circle(image, tuple(D1), MARKER_SIZE, COLORS['board1'], -1)
    cv2.putText(image, f"{D1[0], D1[1]}", (D1[0]-150, D1[1]+400), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['board1'], TEXT_THICKNESS)
    cv2.putText(image, f"Board 1: {distance1:.1f}cm", (D1[0]-150, D1[1]+475), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['board1'], TEXT_THICKNESS)

# Найти вторую шахматную доску
mask = np.ones_like(gray) * 255
if found1:
    cv2.fillPoly(mask, [np.int32(corners1[:,0])], 0)
gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

found2, corners2 = cv2.findChessboardCorners(gray_masked, chessboard_size, flags)
rvec2 = tvec2 = None
if found2:
    corners2 = cv2.cornerSubPix(gray_masked, corners2, (11, 11), (-1, -1), criteria)
    undistorted_corners2 = undistort_points(corners2)
    
    # Рассчитываем 3D-позицию второй доски
    ret2, rvec2, tvec2 = cv2.solvePnP(objp, undistorted_corners2, camera_matrix, camera_params['dist_coeffs'])
    
    # Матрица вращения для второй доски
    Rvec2=cv2.Rodrigues(rvec2)[0]
    #print(Rvec2)

    # Расчет расстояния через tvec
    distance2 = calculate_distance_tvec(tvec2)
    dist2, center2 = calculate_distance(undistorted_corners2, chessboard_size, square_size, camera_params)
    
    # Проектируем 3D-центр доски обратно в 2D
    board_center_3d = np.array([[(chessboard_size[0] - 1) * square_size / 2, 
                                 (chessboard_size[1] - 1) * square_size / 2, 
                                 0]], dtype=np.float32)
    projected_center2, _ = cv2.projectPoints(board_center_3d, rvec2, tvec2, camera_matrix, camera_params['dist_coeffs'])
    D2 = projected_center2.reshape(-1, 2).astype(int)[0]
    
    # Отрисовка 2D-центра
    cv2.circle(image, tuple(D2), MARKER_SIZE, COLORS['board2'], -1)
    cv2.putText(image, f"{D2[0], D2[1]}", (D2[0]-150, D2[1]-400), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['board2'], TEXT_THICKNESS)
    cv2.putText(image, f"Board 2: {distance2:.1f}cm", (D2[0]-150, D2[1]-475), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['board2'], TEXT_THICKNESS)
    
    
    

# Отображение 3D-данных в углу изображения
text_offset_x, text_offset_y = 50, 50
line_spacing = 50

if found1:
    # Форматируем tvec для первой доски
    tvec_str1 = f"Board 1 tvec: [{tvec1[0][0]:.2f}, {tvec1[1][0]:.2f}, {tvec1[2][0]:.2f}]"
    cv2.putText(image, tvec_str1, (text_offset_x, text_offset_y), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    
    rvec_str1 = f"Board 1 rvec: [{rvec1[0][0]:.2f}, {rvec1[1][0]:.2f}, {rvec1[2][0]:.2f}]"
    cv2.putText(image, rvec_str1, (text_offset_x, text_offset_y+75), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    
    cv2.putText(image, f"dist_Board1: {dist1:.1f} cm", (text_offset_x, text_offset_y + 750), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)

if found2:
    # Форматируем tvec для второй доски
    tvec_str2 = f"Board 2 tvec: [{tvec2[0][0]:.2f}, {tvec2[1][0]:.2f}, {tvec2[2][0]:.2f}]"
    cv2.putText(image, tvec_str2, (text_offset_x, text_offset_y + 150), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    
    rvec_str2 = f"Board 2 rvec: [{rvec2[0][0]:.2f}, {rvec2[1][0]:.2f}, {rvec2[2][0]:.2f}]"
    cv2.putText(image, rvec_str2, (text_offset_x, text_offset_y + 225), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    

    cv2.putText(image, f"dist_Board2: {dist2:.1f} cm", (text_offset_x, text_offset_y + 825), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)

# Обнаружение кружки
model = YOLO("yolov8l.pt")
results = model(image)

K=(camera_params['cx'], camera_params['fy'])
K11=(0, 0, 0)

for result in results:
    for box in result.boxes:
        if int(box.cls[0]) == 41:  # Класс "cup"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            O = ((x1 + x2) // 2, (y1 + y2) // 2)
            undistorted_obj_center = undistort_points([O])[0]
            O = (int(undistorted_obj_center[0]), int(undistorted_obj_center[1]))

            # Визуализация объекта
            cv2.rectangle(image, (x1, y1), (x2, y2), COLORS['object'], LINE_THICKNESS)
            cv2.circle(image, O, MARKER_SIZE, COLORS['object'], -1)
            cv2.putText(image, f"Cup", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['object'], TEXT_THICKNESS)
            cv2.putText(image, f"({O[0]},{O[1]})", 
               (O[0], O[1]-80),
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['object'], TEXT_THICKNESS)

if found1 and found2 and 'O' in locals():

    # Расчет углов в 2D-пространстве
    s1 = calculate_angle_s1(D1, O, K)
    #print(f"s1:{s1}")
    s2 = calculate_angle_s2(D2, O, K)
    #print(f"s2:{s2}")
    s3 = calculate_angle_s3(D1, O, K)
    #print(f"s3:{s3}")
    s4 = calculate_angle_s4(D2, O, K) 
    #print(f"s4:{s4}")  
    s5 = calculate_angle_s5(D1, O, K)
    #print(f"s5:{s5}")
    s6 = calculate_angle_s6(D2, O, K)
    #print(f"s6:{s6}")
    # s7 = calculate_angle_s7(D11, K11, D22)
    
    
   # Расчет по теореме синусов
    OK11 = (distance1 * np.sin(np.radians(s1[0])) / np.sin(np.radians(s5[0])))
    OK21 = (distance2 * np.sin(np.radians(s2[0])) / np.sin(np.radians(s6[0])))

    #print(f"OK11: {OK11}")
    #print(f"OK21: {OK21}")

    # Расчет через соотношения 
    OK12 = (distance1*s3[2]/s3[1])
    OK22 = (distance2*s4[1]/s4[2])

    #print(f"OK12: {OK12}")
    #print(f"OK22: {OK22}")

    # Расчет через теорму косинусов
    OK13 = (math.sqrt(s1[1]**2+distance1**2-2*s1[1]*distance1*np.cos(np.radians(s1[0]))))
    OK23 = (math.sqrt(s2[1]**2+distance2**2-2*s2[1]*distance2*np.cos(np.radians(s2[0]))))

    #print(f"OK13: {OK13}")
    #print(f"OK23: {OK23}")

    # Отображение информации о расстоянии 
    
    cv2.putText(image, f"OK11: {OK11:.1f}", (text_offset_x, text_offset_y + 300), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    cv2.putText(image, f"OK21: {OK21:.1f}", (text_offset_x, text_offset_y + 375), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    cv2.putText(image, f"OK12: {OK12:.1f}", (text_offset_x, text_offset_y + 450), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    cv2.putText(image, f"OK22: {OK22:.1f}", (text_offset_x, text_offset_y + 525), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    cv2.putText(image, f"OK13: {OK13:.1f}", (text_offset_x, text_offset_y + 600), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    cv2.putText(image, f"OK23: {OK23:.1f}", (text_offset_x, text_offset_y + 675), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['text'], TEXT_THICKNESS)
    
    

# Масштабирование изображения
scale_factor = min(MAX_DISPLAY_HEIGHT / orig_height, 1.0)
display_width = int(orig_width * scale_factor)
display_height = int(orig_height * scale_factor)

resized_image = cv2.resize(image, (display_width, display_height), 
                         interpolation=cv2.INTER_AREA)

# Отображение результата
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.imshow('Result', resized_image)
cv2.resizeWindow('Result', display_width, display_height)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохранение результатов
cv2.imwrite('full_res_result.jpg', image)
cv2.imwrite('display_result.jpg', resized_image)