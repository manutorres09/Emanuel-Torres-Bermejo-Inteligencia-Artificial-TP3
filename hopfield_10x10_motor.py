# Modelo: Hopfield + Hebb (no pseudoinversa).

import os
import numpy as np

# Utilidades de carga y formato
def load_binary_pattern_txt(path, expected_size=(10,10)):
    """Carga patrón binario {0,1} desde archivo .txt."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ' ' in line:
                vals = [int(x) for x in line.split()]
            else:
                vals = [int(ch) for ch in line]
            rows.append(vals)
    arr = np.array(rows, dtype=int)
    if arr.shape != expected_size:
        raise ValueError(f"Patrón {path} tiene forma {arr.shape}, esperado {expected_size}.")
    return arr

def to_bipolar(arr01):
    """{0,1} -> {-1,+1}."""
    return np.where(arr01==1, 1, -1).astype(np.int8)

def to_binary(arrpm1):
    """{-1,+1} -> {0,1}."""
    return np.where(arrpm1==1, 1, 0).astype(np.int8)

def flatten(arr):
    """Aplana 10x10 -> (100,)"""
    return arr.reshape(-1)

def unflatten(vec, shape=(10,10)):
    return vec.reshape(shape)


# Entrenamiento (Hebb)
def hebb_train(patterns_pm1):
    #Entrena matriz de pesos W con regla de Hebb.
    N = patterns_pm1[0].size
    W = np.zeros((N, N), dtype=np.int32)
    for x in patterns_pm1:
        x = x.reshape(-1,1)
        W += (x @ x.T)
    np.fill_diagonal(W, 0)
    return W


# Dinámica de recuperación

def sign_func(v):
    # Convención: sgn(0) -> +1
    return np.where(v>=0, 1, -1).astype(np.int8)

def recall(W, y0, max_iters=60, mode='async'):
    # Recupera patrón desde y0 usando pesos W.
    y = y0.copy()
    N = y.size
    energies = [hopfield_energy(W, y)]
    for it in range(1, max_iters+1):
        if mode == 'sync':
            y_new = sign_func(W @ y)
        else:
            y_new = y.copy()
            for i in range(N):
                h_i = int(np.dot(W[i], y_new))
                y_new[i] = 1 if h_i >= 0 else -1
        energies.append(hopfield_energy(W, y_new))
        if np.array_equal(y_new, y):
            return y_new, energies, it, True
        y = y_new
    return y, energies, max_iters, False

def hopfield_energy(W, state):
    # E = -0.5 * state^T W state
    return -0.5 * float(state.T @ (W @ state))

# Ruido / distorsión
def flip_bits(vec_pm1, flip_ratio=0.2, rng=None):
    """Invierte aleatoriamente flip_ratio de las posiciones."""
    rng = np.random.default_rng(rng)
    y = vec_pm1.copy()
    N = y.size
    k = int(np.round(flip_ratio * N))
    if k > 0:
        idx = rng.choice(N, size=k, replace=False)
        y[idx] *= -1
    return y

# Visualización en consola (ASCII)
def print_pattern_ascii(arr01):
    """1 -> '█' ; 0 -> '·'"""
    chars = {1:'█', 0:'·'}
    for row in arr01:
        print(''.join(chars[int(v)] for v in row))

# Main
if __name__ == "__main__":
    folder = "patrones"
    if not os.path.exists(folder):
        print("No se encontró la carpeta 'patrones'. Creala y coloca tus archivos .txt de 10x10 (0/1).")
        exit(1)

    # Buscar patrones
    pattern_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
    pattern_files = sorted(pattern_files)

    if len(pattern_files) < 4:
        print("Se requieren al menos 4 archivos .txt en la carpeta 'patrones' (cada uno 10x10 con 0 y 1).")
        exit(1)

    # 1) Cargar patrones desde archivos
    patterns01 = [load_binary_pattern_txt(p) for p in pattern_files]

    # 2) Convertir a bipolar y aplanar
    patterns_pm1 = [flatten(to_bipolar(p)) for p in patterns01]

    # 3) Entrenar con Hebb
    W = hebb_train(patterns_pm1)

    # 4) Elegir un patrón objetivo y crear versión ruidosa
    target_idx = 0
    x_clean = patterns_pm1[target_idx]
    x_noisy = flip_bits(x_clean, flip_ratio=0.25, rng=123)  # 25% ruido

    # 5) Recuperación
    y_rec, E_hist, iters, ok = recall(W, x_noisy, max_iters=60, mode='async')

    # 6) Mostrar resultados
    clean_img = to_binary(unflatten(x_clean))
    noisy_img = to_binary(unflatten(x_noisy))
    rec_img   = to_binary(unflatten(y_rec))

    print("\n== Patrón limpio (objetivo) ==")
    print_pattern_ascii(clean_img)
    print("\n== Patrón de entrada (ruidoso) ==")
    print_pattern_ascii(noisy_img)
    print("\n== Patrón recuperado ==")
    print_pattern_ascii(rec_img)

    print(f"\nEnergía inicial: {E_hist[0]:.1f} -> final: {E_hist[-1]:.1f} | iteraciones: {iters} | convergió: {ok}")

    # 7) Evaluar similitud con patrones entrenados
    def match_score(a_pm1, b_pm1):
        return (a_pm1 == b_pm1).mean()

    scores = [match_score(y_rec, p) for p in patterns_pm1]
    best = int(np.argmax(scores))
    print("\nScores de coincidencia con cada patrón entrenado:")
    for i, sc in enumerate(scores):
        print(f"  Patrón {i+1}: {sc*100:.1f}%")
    print(f"\nClasificación (mejor coincidencia): patrón {best+1}")
