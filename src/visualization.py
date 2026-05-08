import numpy as np
import matplotlib.pyplot as plt
from .constants import GRID_LIMIT, GRID_SIZE
from .physics import calculate_ppm_map

def plot_2d_slice(field, title="Срез поля"):
    """Отрисовка 2D тепловой карты (срез центральной плоскости XY)."""
    # Берем центральный срез Z
    z_idx = GRID_SIZE // 2
    slice_field = field[:, :, z_idx, :]
    
    # Переводим в PPM
    ppm_map = calculate_ppm_map(slice_field)
    
    # Координаты для осей (в мм)
    extent =[-GRID_LIMIT*1000, GRID_LIMIT*1000, -GRID_LIMIT*1000, GRID_LIMIT*1000]
    
    plt.figure(figsize=(7, 6))
    # cmap='seismic' идеально показывает + (красный) и - (синий) отклонения
    img = plt.imshow(ppm_map.T, extent=extent, origin='lower', cmap='seismic')
    
    plt.colorbar(img, label='Отклонение поля (ppm)')
    plt.title(title, fontsize=14)
    plt.xlabel('X, мм', fontsize=12)
    plt.ylabel('Y, мм', fontsize=12)
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_coils_3d(coils_config):
    """Схематичная 3D визуализация расположения катушек."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Рисуем "рабочий куб" (границы оптимизации)
    lim = GRID_LIMIT * 1000  # в мм
    r = [-lim, lim]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="green", alpha=0.2)

    # Рисуем катушки (как точки и векторы их направления)
    for idx, coil in enumerate(coils_config):
        pos = np.array(coil["pos"]) * 1000  # м -> мм
        axis = np.array(coil["axis"])
        
        # Точка центра
        ax.scatter(pos[0], pos[1], pos[2], color='red', s=50)
        # Вектор нормали
        ax.quiver(pos[0], pos[1], pos[2], 
                  axis[0], axis[1], axis[2], 
                  length=10, color='blue', normalize=True)
        # Номер катушки
        ax.text(pos[0], pos[1], pos[2], f'  {idx+1}', color='black')
        
    ax.set_title('Расположение шиммирующих катушек и область сканирования')
    ax.set_xlabel('X, мм')
    ax.set_ylabel('Y, мм')
    ax.set_zlabel('Z, мм')
    
    # Выравниваем масштаб осей
    ax.set_box_aspect([1,1,1]) 
    plt.show()

# Импорты для рисования куба
from itertools import product, combinations