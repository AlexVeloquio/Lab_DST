"""
Geometria pura sobre arreglos de landmarks (N, 3). Sin mediapipe, sin cv2.

La idea central del proyecto: MediaPipe entrega PUNTOS, no gestos. Todo el
valor de ingenieria esta en convertir esos puntos en magnitudes que NO dependan
de la distancia a la camara, de la rotacion de la mano ni de si es izquierda o
derecha. Eso es lo que hace este modulo.
"""

from __future__ import annotations

import numpy as np

from . import landmarks as LM


# ---------------------------------------------------------------------------
# Utilidades vectoriales
# ---------------------------------------------------------------------------
def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normaliza un vector (o filas de una matriz) a norma 1."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angulo ABC en grados (vertice en b). Funciona en 2D o 3D.
    Es la primitiva mas usada: flexion de una articulacion.
    """
    v1 = unit(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    v2 = unit(np.asarray(c, dtype=float) - np.asarray(b, dtype=float))
    cos = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def palm_size(pts: np.ndarray) -> float:
    """
    Escala de referencia de la mano: distancia muneca -> MCP del dedo medio.
    Se usa como denominador para volver TODA distancia adimensional; asi un
    gesto se detecta igual a 50 cm que a 3 m de la camara.
    """
    d = np.linalg.norm(pts[LM.MIDDLE_MCP] - pts[LM.WRIST])
    return float(max(d, 1e-6))


def normalized_distance(pts: np.ndarray, i: int, j: int) -> float:
    """Distancia entre dos landmarks expresada en 'palmas'."""
    return float(np.linalg.norm(pts[i] - pts[j]) / palm_size(pts))


# ---------------------------------------------------------------------------
# Marco canonico de la mano
# ---------------------------------------------------------------------------
def canonical_basis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye una base ortonormal anclada a la mano.

      origen : muneca (landmark 0)
      eje Y  : muneca -> MCP medio   (direccion "hacia los dedos")
      eje X  : MCP indice -> MCP menique, ortogonalizado contra Y (ancho palma)
      eje Z  : X x Y (normal de la palma -> indica si vemos palma o dorso)

    Devuelve (origen, R) con R de forma (3, 3), filas = [x, y, z].
    """
    origin = pts[LM.WRIST].astype(float)

    y = unit(pts[LM.MIDDLE_MCP] - origin)
    x_raw = pts[LM.PINKY_MCP] - pts[LM.INDEX_MCP]
    # Gram-Schmidt: quitamos de x la componente paralela a y
    x = unit(x_raw - np.dot(x_raw, y) * y)
    z = np.cross(x, y)

    return origin, np.stack([x, y, z])


def to_canonical(pts: np.ndarray) -> np.ndarray:
    """
    Lleva los 21 puntos al marco canonico y los escala por el tamano de palma.
    Resultado: invariante a traslacion, rotacion y escala.
    Es la representacion correcta para comparar posturas o alimentar un
    clasificador entrenado.
    """
    origin, R = canonical_basis(pts)
    return (pts.astype(float) - origin) @ R.T / palm_size(pts)


def palm_normal_sign(pts: np.ndarray, handedness: str) -> float:
    """
    +1 si vemos la PALMA hacia la camara, -1 si vemos el DORSO.
    MediaPipe entrega z negativo hacia la camara; el producto cruz cambia de
    signo segun la mano, por eso se corrige con handedness.
    """
    _, R = canonical_basis(pts)
    z = R[2]
    s = 1.0 if z[2] < 0 else -1.0
    return s if handedness.lower().startswith(("r", "d")) else -s


def mirror_x(pts: np.ndarray) -> np.ndarray:
    """
    Espeja en X. Permite tratar mano izquierda y derecha con el MISMO conjunto
    de reglas geometricas: se espeja la izquierda y se evalua como derecha.
    """
    out = pts.copy().astype(float)
    out[:, 0] = -out[:, 0]
    return out


def centroid(pts: np.ndarray) -> np.ndarray:
    return pts.astype(float).mean(axis=0)


def sample_bone(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Interpola n puntos sobre el segmento p0-p1 (para 'rellenar' el esqueleto)."""
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return p0.reshape(1, -1) * (1 - t) + p1.reshape(1, -1) * t


def densify(pts: np.ndarray, per_bone: int = 6) -> np.ndarray:
    """
    Convierte los 21 puntos en una nube densa siguiendo los huesos.
    Necesario para el mapeo a los 16 modulos: si solo se usan 21 puntos, las
    celdas de la matriz quedan huecas y el 'reflejo' se ve punteado.
    """
    chunks = [pts.astype(float)]
    for a, b in LM.HAND_CONNECTIONS:
        chunks.append(sample_bone(pts[a], pts[b], per_bone))
    return np.vstack(chunks)
