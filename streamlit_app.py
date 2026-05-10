import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time

# Импортируем вашу логику
from src.constants import GRID_LIMIT, MAX_CURRENT
from src.physics import get_grid, generate_main_field, calculate_ppm_std, bio_savart
from src.optimization import run_optimization
from src.coils_config import coils_config

# Настройка страницы
st.set_page_config(page_title="MRI Shimming Lab", layout="wide")

st.title("Лаборатория активного шимминга МРТ")
st.markdown("Перемещайте катушки и выбирайте алгоритмы, чтобы увидеть, как система борется с неоднородностью поля.")

# --- SIDEBAR (Панель управления) ---
st.sidebar.header("Настройки системы")

# 1. Выбор алгоритма
method = st.sidebar.selectbox(
    "Алгоритм оптимизации",
    ["L-BFGS-B", "Nelder-Mead", "Differential Evolution", "Dual Annealing"]
)

# 2. Управление катушками
st.sidebar.subheader("Положение катушек")
coil_idx = st.sidebar.number_input("Выберите номер катушки (1-12)", 1, 12, 1) - 1

# Слайдеры для изменения координат выбранной катушки
# Мы позволяем двигать катушку в диапазоне ±20 мм от её начальной позиции
current_pos = coils_config[coil_idx]["pos"]
new_x = st.sidebar.slider("Смещение по X (м)", -0.1, 0.1, float(current_pos[0]), step=0.005)
new_y = st.sidebar.slider("Смещение по Y (м)", -0.1, 0.1, float(current_pos[1]), step=0.005)
new_z = st.sidebar.slider("Смещение по Z (м)", -0.1, 0.1, float(current_pos[2]), step=0.005)

# Обновляем конфиг в памяти приложения
coils_config[coil_idx]["pos"] = [new_x, new_y, new_z]

# --- LOGIC (Расчеты) ---
@st.cache_data # Кэшируем сетку и исходное поле, чтобы не пересчитывать при каждом движении слайдера
def load_base_data():
    grid = get_grid()
    B_main = generate_main_field(grid)
    return grid, B_main

grid, B_main = load_base_data()

# Кнопка запуска (для тяжелых алгоритмов лучше запускать по кнопке, для L-BFGS-B можно в реальном времени)
run_auto = method == "L-BFGS-B" or method == "Nelder-Mead"

if run_auto or st.sidebar.button("Рассчитать шимминг"):
    
    with st.spinner('Математика в процессе...'):
        # 1. Пересчитываем единичные поля (так как геометрия могла измениться)
        unit_fields = []
        for c in coils_config:
            # Используем n_points=16 для скорости в интерактивном режиме
            B_u = bio_savart(c["pos"], c["axis"], 1.0, grid, n_points=16)
            unit_fields.append(B_u)
        unit_fields = np.array(unit_fields)

        # 2. Оптимизация
        start_t = time.time()
        opt_currents, final_ppm = run_optimization(B_main, unit_fields, method)
        exec_time = time.time() - start_t

        # 3. Итоговое поле
        B_comp = B_main + np.tensordot(opt_currents, unit_fields, axes=1)
        initial_ppm = calculate_ppm_std(B_main)

    # --- VISUALIZATION (Отрисовка) ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Результаты")
        res_df = pd.DataFrame({
            "Параметр": ["Исходный PPM", "Итоговый PPM", "Улучшение (раз)", "Время счета"],
            "Значение": [
                f"{initial_ppm:.2f}", 
                f"{final_ppm:.2f}", 
                f"{initial_ppm/final_ppm:.1f}x",
                f"{exec_time:.3f} сек"
            ]
        })
        st.table(res_df)
        
        st.subheader("Оптимальные токи (А)")
        st.bar_chart(opt_currents)

    with col2:
        st.subheader("Карта неоднородности (Срез XY)")
        
        # Подготовка данных для Plotly Heatmap
        z_slice = grid.shape[2] // 2
        # Считаем отклонение в PPM для каждой точки среза
        B_norm = np.linalg.norm(B_comp, axis=-1)[:, :, z_slice]
        mean_B = np.mean(np.linalg.norm(B_comp, axis=-1))
        ppm_map = (B_norm - mean_B) / mean_B * 1e6

        fig = go.Figure(data=go.Heatmap(
            z=ppm_map.T,
            x=np.linspace(-GRID_LIMIT*1000, GRID_LIMIT*1000, ppm_map.shape[0]),
            y=np.linspace(-GRID_LIMIT*1000, GRID_LIMIT*1000, ppm_map.shape[1]),
            colorscale='RdBu_r',
            zmid=0
        ))
        fig.update_layout(xaxis_title="X (мм)", yaxis_title="Y (мм)", width=500, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # 3D Вид катушек
    st.subheader("3D Визуализация системы")
    fig_3d = go.Figure()
    
    # Рисуем катушки
    for i, c in enumerate(coils_config):
        color = 'red' if i == coil_idx else 'blue'
        fig_3d.add_trace(go.Scatter3d(
            x=[c["pos"][0]], y=[c["pos"][1]], z=[c["pos"][2]],
            mode='markers+text',
            marker=dict(size=8, color=color),
            text=[f"C{i+1}"],
            name=f"Катушка {i+1}"
        ))

    fig_3d.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='cube'
    ), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("Нажмите кнопку 'Рассчитать шимминг' или выберите быстрый алгоритм (L-BFGS-B), чтобы увидеть результат.")