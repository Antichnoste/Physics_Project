import numpy as np
from .constants import MU0, GRID_SIZE, GRID_LIMIT, COIL_TURNS, COIL_RADIUS, MAIN_B0

def get_grid():
    """Создает 3D сетку координат для расчета."""
    x = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    y = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    z = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.stack((X, Y, Z), axis=-1)

def generate_main_field(grid):
    """Генерация неоднородного главного поля B0 (Имитация реального МРТ)."""
    X, Y, Z = grid[..., 0], grid[..., 1], grid[..., 2]
    
    # Главное поле направлено по оси Z. Добавляем искажения (порядка сотен ppm)
    Bz = MAIN_B0 + 5e-4 * np.exp(-(Z**2 + X**2) / (2 * (0.015**2))) + 2e-4 * np.sin(50 * Y)
    Bx = 1e-4 * np.sin(40 * X)
    By = 1e-4 * np.cos(40 * Y)
    
    return np.stack((Bx, By, Bz), axis=-1)

def bio_savart(coil_pos, coil_normal, current, points, n_points=32):
    """Точный расчет магнитного поля кругового витка произвольной ориентации."""
    coil_pos = np.array(coil_pos)
    coil_normal = np.array(coil_normal)
    coil_normal = coil_normal / np.linalg.norm(coil_normal) # Нормализуем ось
    
    # 1. Находим локальные векторы в плоскости катушки (u и v)
    if np.allclose(np.abs(coil_normal), [0, 0, 1]):
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.cross([0.0, 0.0, 1.0], coil_normal)
        u = u / np.linalg.norm(u)
    v = np.cross(coil_normal, u)
    
    # 2. Разбиваем кольцо на отрезки
    theta = np.linspace(0, 2 * np.pi, n_points + 1)
    # Координаты точек кольца в пространстве
    ring_points = coil_pos + COIL_RADIUS * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    
    # 3. Численное интегрирование по закону Био-Савара
    B = np.zeros_like(points)
    points_flat = points.reshape(-1, 3)
    B_flat = np.zeros_like(points_flat)
    
    for i in range(n_points):
        p_start = ring_points[i]
        p_end = ring_points[i+1]
        dl = p_end - p_start
        p_mid = (p_start + p_end) / 2.0
        
        r = points_flat - p_mid
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        # Защита от деления на ноль, если точка попала прямо на провод
        r_norm = np.where(r_norm < 1e-6, 1e-6, r_norm) 
        
        # Интеграл: (dl x r) / |r|^3
        B_flat += np.cross(dl, r) / (r_norm**3)
        
    B_total_flat = (MU0 * current * COIL_TURNS / (4 * np.pi)) * B_flat
    return B_total_flat.reshape(points.shape)

def calculate_ppm_std(field):
    """Считает общую неоднородность поля (RMS ppm) — это наша целевая функция."""
    B_norm = np.linalg.norm(field, axis=-1)
    mean_B = np.mean(B_norm)
    std_B = np.std(B_norm)
    return (std_B / mean_B) * 1e6

def calculate_ppm_map(field):
    """Считает локальную неоднородность в каждой точке для графиков (ppm)."""
    B_norm = np.linalg.norm(field, axis=-1)
    mean_B = np.mean(B_norm)
    return (B_norm - mean_B) / mean_B * 1e6