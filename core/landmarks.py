"""
Constantes de topologia de MediaPipe (manos y cuerpo).

IMPORTANTE: desde mediapipe 1.0 el modulo `mediapipe.solutions` fue eliminado,
por lo que `mp.solutions.hands.HAND_CONNECTIONS` y `drawing_utils` ya NO estan
disponibles. Este modulo replica esas constantes para que el dibujo y la
geometria no dependan de la API legacy.

Este archivo es PURO (no importa mediapipe ni cv2) para poder probarlo offline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MANO: 21 landmarks (indices oficiales de MediaPipe Hand Landmarker)
# ---------------------------------------------------------------------------
WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_LANDMARK_NAMES = [
    "MUNECA",
    "PULGAR_CMC", "PULGAR_MCP", "PULGAR_IP", "PULGAR_PUNTA",
    "INDICE_MCP", "INDICE_PIP", "INDICE_DIP", "INDICE_PUNTA",
    "MEDIO_MCP", "MEDIO_PIP", "MEDIO_DIP", "MEDIO_PUNTA",
    "ANULAR_MCP", "ANULAR_PIP", "ANULAR_DIP", "ANULAR_PUNTA",
    "MENIQUE_MCP", "MENIQUE_PIP", "MENIQUE_DIP", "MENIQUE_PUNTA",
]

FINGER_NAMES = ["pulgar", "indice", "medio", "anular", "menique"]

# Cadena cinematica de cada dedo: (raiz, MCP, PIP/IP, DIP, PUNTA)
# El pulgar tiene una articulacion menos util, se repite el patron de 4 puntos.
FINGER_CHAINS = {
    "pulgar":  [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    "indice":  [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    "medio":   [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    "anular":  [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    "menique": [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
}

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Conexiones para dibujar el esqueleto de la mano (reemplaza HAND_CONNECTIONS)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),            # indice
    (9, 10), (10, 11), (11, 12),               # medio
    (13, 14), (14, 15), (15, 16),              # anular
    (0, 17), (17, 18), (18, 19), (19, 20),     # menique
    (5, 9), (9, 13), (13, 17),                 # arco de nudillos (palma)
)

# Colores BGR por dedo, para depuracion visual
FINGER_COLORS = {
    "pulgar":  (80, 80, 255),
    "indice":  (80, 200, 255),
    "medio":   (80, 255, 120),
    "anular":  (255, 200, 80),
    "menique": (255, 120, 220),
}

# ---------------------------------------------------------------------------
# CUERPO: 33 landmarks (Pose Landmarker)
# ---------------------------------------------------------------------------
POSE_NOSE = 0
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW = 13, 14
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
POSE_LEFT_HIP, POSE_RIGHT_HIP = 23, 24

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
