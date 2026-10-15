"""
Extraccion de caracteristicas de la mano: "que estan haciendo los dedos".

Cada dedo se resume en un escalar continuo de extension en [0, 1]:
    0.0 = completamente flexionado (cerrado)
    1.0 = completamente extendido

Se combinan DOS evidencias independientes, porque cada una falla en un caso
distinto:

  (a) ANGULOS de flexion (PIP + DIP). Robustos ante escala y perspectiva, pero
      degeneran cuando el dedo apunta hacia la camara (escorzo): los tres
      puntos se proyectan casi encima y el angulo se vuelve ruido.

  (b) RAZON DE DISTANCIAS  d(punta, muneca) / d(MCP, muneca). Sobrevive al
      escorzo, pero se confunde cuando la mano gira de perfil.

La fusion ponderada de ambas da una senal mucho mas estable que cualquiera de
las dos por separado. Este es el tipo de argumento que conviene documentar en
el reporte del laboratorio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as G
from . import landmarks as LM


# Rangos empiricos de angulo para mapear a [0, 1].
ANGLE_CLOSED_DEG = 95.0    # articulacion muy doblada
ANGLE_OPEN_DEG = 172.0     # articulacion recta

# El pulgar nunca llega a 180 grados: necesita su propio rango.
THUMB_CLOSED_DEG = 120.0
THUMB_OPEN_DEG = 168.0

# Razon de distancias punta/MCP respecto a la muneca.
RATIO_CLOSED = 0.85
RATIO_OPEN = 1.55


def _lerp01(x: float, lo: float, hi: float) -> float:
    """Mapea x del rango [lo, hi] al rango [0, 1] con saturacion."""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def finger_extension(pts: np.ndarray, finger: str) -> float:
    """Extension continua [0,1] de un dedo, fusionando angulo y distancia."""
    chain = LM.FINGER_CHAINS[finger]
    mcp, pip, dip, tip = chain

    a1 = G.angle_between(pts[mcp], pts[pip], pts[dip])
    a2 = G.angle_between(pts[pip], pts[dip], pts[tip])

    if finger == "pulgar":
        lo, hi = THUMB_CLOSED_DEG, THUMB_OPEN_DEG
    else:
        lo, hi = ANGLE_CLOSED_DEG, ANGLE_OPEN_DEG

    ext_angle = 0.5 * (_lerp01(a1, lo, hi) + _lerp01(a2, lo, hi))

    d_tip = np.linalg.norm(pts[tip] - pts[LM.WRIST])
    d_mcp = np.linalg.norm(pts[mcp] - pts[LM.WRIST])
    ratio = float(d_tip / max(d_mcp, 1e-6))
    ext_ratio = _lerp01(ratio, RATIO_CLOSED, RATIO_OPEN)

    # El angulo pesa mas porque es el mas informativo cuando la mano se ve
    # frontalmente, que es el caso de uso dominante frente a sOFIa.
    return float(np.clip(0.65 * ext_angle + 0.35 * ext_ratio, 0.0, 1.0))


def finger_curl_angles(pts: np.ndarray, finger: str) -> tuple[float, float]:
    """Devuelve (angulo PIP, angulo DIP) en grados. Util para graficar/depurar."""
    mcp, pip, dip, tip = LM.FINGER_CHAINS[finger]
    return (
        G.angle_between(pts[mcp], pts[pip], pts[dip]),
        G.angle_between(pts[pip], pts[dip], pts[tip]),
    )


def finger_spread(pts: np.ndarray) -> float:
    """
    Apertura lateral entre dedos (abduccion), normalizada por tamano de palma.
    Distingue "mano abierta con dedos juntos" de "mano abierta en estrella".
    """
    tips = [LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP]
    gaps = [
        G.normalized_distance(pts, tips[i], tips[i + 1])
        for i in range(len(tips) - 1)
    ]
    return float(np.mean(gaps))


def pinch_distance(pts: np.ndarray) -> float:
    """Distancia pulgar-indice en palmas. < ~0.35 se considera pinza cerrada."""
    return G.normalized_distance(pts, LM.THUMB_TIP, LM.INDEX_TIP)


def hand_aperture(extensions: dict[str, float]) -> float:
    """Apertura global [0,1]: promedio ponderado de los cinco dedos."""
    w = {"pulgar": 0.8, "indice": 1.0, "medio": 1.0, "anular": 1.0, "menique": 0.9}
    num = sum(extensions[f] * w[f] for f in LM.FINGER_NAMES)
    return float(num / sum(w.values()))


def pointing_direction(pts: np.ndarray) -> np.ndarray:
    """Vector unitario MCP->punta del indice. Define hacia donde senala."""
    return G.unit(pts[LM.INDEX_TIP] - pts[LM.INDEX_MCP])


@dataclass
class HandFeatures:
    """Todo lo que el resto del sistema necesita saber de UNA mano en UN frame."""

    handedness: str                      # "Left" / "Right"
    score: float                         # confianza del clasificador de mano
    points: np.ndarray                   # (21,3) en pixeles o normalizado
    world: np.ndarray                    # (21,3) metrico, origen en la mano
    extensions: dict[str, float] = field(default_factory=dict)
    fingers_up: dict[str, bool] = field(default_factory=dict)
    aperture: float = 0.0
    spread: float = 0.0
    pinch: float = 1.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0                   # tamano de palma EN PIXELES
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    palm_facing: float = 1.0             # +1 palma, -1 dorso (discreto)
    palm_normal: float = 1.0             # [-1,+1] continuo; 0 = mano de canto
    timestamp_ms: int = 0

    @property
    def n_fingers_up(self) -> int:
        return sum(1 for v in self.fingers_up.values() if v)

    @property
    def orientacion(self) -> str:
        """PALMA / DORSO / CANTO. CANTO es la zona donde el signo no es fiable."""
        if abs(self.palm_normal) < 0.35:
            return "CANTO"
        return "PALMA" if self.palm_normal > 0 else "DORSO"

    @property
    def code(self) -> str:
        """Firma binaria de 5 bits, ej. '01100'. Comoda para depurar y loggear."""
        return "".join("1" if self.fingers_up[f] else "0" for f in LM.FINGER_NAMES)


def extract(
    points: np.ndarray,
    world: np.ndarray,
    handedness: str,
    score: float,
    timestamp_ms: int,
    up_threshold: float = 0.55,
) -> HandFeatures:
    """
    Calcula todas las caracteristicas de una mano.

    Los angulos se calculan sobre `world` (coordenadas metricas 3D reales, sin
    distorsion de perspectiva) mientras que la posicion en pantalla se toma de
    `points`. Usar world para geometria es la diferencia entre un detector que
    funciona solo de frente y uno que aguanta que la persona este de lado.
    """
    geo = world if world is not None and len(world) == 21 else points

    ext = {f: finger_extension(geo, f) for f in LM.FINGER_NAMES}
    ups = {f: ext[f] >= up_threshold for f in LM.FINGER_NAMES}

    return HandFeatures(
        handedness=handedness,
        score=score,
        points=points,
        world=geo,
        extensions=ext,
        fingers_up=ups,
        aperture=hand_aperture(ext),
        spread=finger_spread(geo),
        pinch=pinch_distance(geo),
        center=G.centroid(points),
        scale=G.palm_size(points),
        direction=pointing_direction(points),
        palm_normal=G.palm_normal_value(geo, handedness),
        palm_facing=G.palm_normal_sign(geo, handedness),
        timestamp_ms=timestamp_ms,
    )
