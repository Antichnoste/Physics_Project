import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from .constants import MAX_CURRENT
from .physics import bio_savart

def precompute_unit_fields(grid, coils_config):
    """Вычисляет поле от каждой катушки при токе в 1А. Занимает пару секунд 1 раз."""
    unit_fields =[]
    for coil in coils_config:
        B_unit = bio_savart(coil["pos"], coil["axis"], current=1.0, points=grid)
        unit_fields.append(B_unit)
    return np.array(unit_fields)

def run_optimization(B_main, unit_fields, method_name="L-BFGS-B"):
    """
    Запуск выбранного алгоритма оптимизации.
    Благодаря unit_fields расчет происходит мгновенно!
    """
    # Сохраняем исходные формы для быстроты
    flat_B_main = B_main.reshape(-1, 3)
    flat_unit_fields = unit_fields.reshape(len(unit_fields), -1, 3)
    num_coils = len(unit_fields)
    bounds = [(-MAX_CURRENT, MAX_CURRENT)] * num_coils

    # Целевая функция, которую вызываем тысячи раз (теперь она летает)
    def objective(currents):
        # Суперпозиция: B_main + I1*B_unit1 + I2*B_unit2 ...
        B_total = flat_B_main + np.tensordot(currents, flat_unit_fields, axes=1)
        B_norm = np.linalg.norm(B_total, axis=1)
        # Возвращаем PPM (стандартное отклонение / среднее * 1e6)
        return (np.std(B_norm) / np.mean(B_norm)) * 1e6

    # Начальное приближение (все токи равны 0)
    x0 = np.zeros(num_coils)

    if method_name == "Nelder-Mead":
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000})
        
    elif method_name == "L-BFGS-B":
        # Очень быстрый алгоритм для гладких функций
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
    elif method_name == "Differential Evolution":
        # Глобальная оптимизация (медленнее, но надежнее)
        res = differential_evolution(objective, bounds=bounds, seed=42)
        
    elif method_name == "Dual Annealing":
        res = dual_annealing(objective, bounds=bounds, maxiter=200, seed=42)
        
    else:
        raise ValueError(f"Неизвестный метод: {method_name}")

    return res.x, res.fun  # Возвращаем оптимальные токи и достигнутый PPM