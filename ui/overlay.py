"""
Dibujo del esqueleto y HUD con OpenCV puro.

Reemplaza a `mp.solutions.drawing_utils`, que ya no existe. Dibujar a mano no
es un castigo: da control total sobre colores por dedo, grosor proporcional a
la confianza y el HUD de depuracion que pide la semana 9.
"""

from __future__ import annotations

import cv2
import numpy as np

from core import landmarks as LM
from core.features import HandFeatures
from core.mapper import GRID

BG = (24, 24, 28)
FG = (235, 235, 240)
ACCENT = (70, 200, 255)
DIM = (110, 110, 120)


def draw_hand(img: np.ndarray, f: HandFeatures, show_ids: bool = False) -> None:
    """Dibuja huesos, articulaciones y opcionalmente el indice de cada punto."""
    pts = f.points[:, :2].astype(int)

    for a, b in LM.HAND_CONNECTIONS:
        color = DIM
        for finger, chain in LM.FINGER_CHAINS.items():
            if a in chain and b in chain:
                color = LM.FINGER_COLORS[finger]
                if not f.fingers_up[finger]:
                    color = tuple(int(c * 0.35) for c in color)
                break
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        r = 5 if i in LM.FINGER_TIPS else 3
        cv2.circle(img, (x, y), r, FG, -1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(img, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, ACCENT, 1, cv2.LINE_AA)

    wx, wy = pts[LM.WRIST]
    cv2.putText(img, f"{f.handedness} {f.code}", (wx - 30, wy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)

    # Orientacion de la palma: es la señal que alimenta el gesto GIRO, asi que
    # verla en vivo es lo que permite calibrar el umbral de la banda de canto.
    orient = f.orientacion
    color = {"PALMA": (90, 220, 120), "DORSO": (90, 160, 255),
             "CANTO": (120, 120, 130)}[orient]
    cv2.putText(img, f"{orient} {f.palm_normal:+.2f}", (wx - 30, wy + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_pose(img: np.ndarray, pose: np.ndarray) -> None:
    pts = pose[:, :2].astype(int)
    for a, b in LM.POSE_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (90, 90, 100), 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 2, (150, 150, 160), -1, cv2.LINE_AA)


def draw_palm_gauge(img: np.ndarray, f: HandFeatures,
                    origin: tuple[int, int] = (12, 202), width: int = 160) -> None:
    """
    Barra deslizante de orientacion de palma, con la banda de canto marcada.

    Sirve para calibrar el gesto GIRO: la zona gris central es donde el signo
    de la normal no es confiable y el detector se niega a comprometerse.
    """
    x0, y0 = origin
    cv2.putText(img, "PALMA <-> DORSO", (x0, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, FG, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + 12), (60, 60, 70), -1)

    # banda muerta (|v| < 0.5), donde el detector considera "de canto"
    c = x0 + width // 2
    half = int(width * 0.25)
    cv2.rectangle(img, (c - half, y0), (c + half, y0 + 12), (95, 95, 105), -1)

    pos = int(x0 + width * (1.0 - (f.palm_normal + 1.0) / 2.0))
    color = {"PALMA": (90, 220, 120), "DORSO": (90, 160, 255),
             "CANTO": (150, 150, 160)}[f.orientacion]
    cv2.rectangle(img, (pos - 2, y0 - 2), (pos + 2, y0 + 14), color, -1)


def draw_finger_bars(img: np.ndarray, f: HandFeatures,
                     origin: tuple[int, int] = (12, 90)) -> None:
    """Barras de extension por dedo. Es la grafica que mas sirve al calibrar."""
    x0, y0 = origin
    for i, finger in enumerate(LM.FINGER_NAMES):
        y = y0 + i * 20
        ext = f.extensions[finger]
        cv2.putText(img, finger[:4].upper(), (x0, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FG, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0 + 45, y), (x0 + 165, y + 12), (60, 60, 70), -1)
        cv2.rectangle(img, (x0 + 45, y), (x0 + 45 + int(120 * ext), y + 12),
                      LM.FINGER_COLORS[finger], -1)
        cv2.putText(img, f"{ext:.2f}", (x0 + 172, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)


def draw_matrix(img: np.ndarray, values: np.ndarray,
                origin: tuple[int, int] = (12, 240), cell: int = 34) -> None:
    """
    Previsualiza los 16 modulos como una matriz 4x4.
    Permite validar el mapeo SIN hardware conectado, que es lo que desbloquea
    trabajar en paralelo con el equipo de mecatronica.
    """
    x0, y0 = origin
    cv2.putText(img, "sOFIa 4x4", (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, FG, 1, cv2.LINE_AA)
    for i, v in enumerate(values):
        r, c = divmod(i, GRID)
        x, y = x0 + c * cell, y0 + r * cell
        t = float(np.clip(v / 255.0, 0, 1))
        color = (int(40 + 60 * t), int(40 + 160 * t), int(50 + 205 * t))
        cv2.rectangle(img, (x, y), (x + cell - 4, y + cell - 4), color, -1)
        cv2.rectangle(img, (x, y), (x + cell - 4, y + cell - 4), (70, 70, 80), 1)
        cv2.putText(img, f"{int(v)}", (x + 4, y + cell - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (20, 20, 20) if t > 0.5 else (170, 170, 180), 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, fps: float, n_hands: int, n_poses: int,
             gesture: str, link_ok: bool, strategy: str) -> None:
    """Franja superior con el estado del sistema completo."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 34), BG, -1)

    cv2.putText(img, f"FPS {fps:5.1f}", (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, FG, 1, cv2.LINE_AA)
    cv2.putText(img, f"manos {n_hands}  personas {n_poses}", (110, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, FG, 1, cv2.LINE_AA)
    cv2.putText(img, f"mapeo {strategy}", (300, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)

    est = "ESP32 OK" if link_ok else "ESP32 --"
    cv2.putText(img, est, (w - 210, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (90, 220, 120) if link_ok else (90, 90, 220), 1, cv2.LINE_AA)

    if gesture and gesture != "NINGUNO":
        cv2.putText(img, gesture, (w - 110, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)


def draw_banner(img: np.ndarray, text: str, color=(60, 60, 200)) -> None:
    """Aviso grande: camara perdida, enlace caido, etc."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 40), (w, h), color, -1)
    cv2.putText(img, text, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
