"""
Envoltorio de MediaPipe con SOPORTE PARA LAS DOS APIs.

Este archivo es el unico del proyecto que importa mediapipe. Detecta solo cual
API esta disponible y se adapta:

  BACKEND "legacy"  ->  mp.solutions.hands / mp.solutions.pose
                        Disponible en mediapipe <= 0.10.21 (Python 3.8-3.11).
                        Es la API de los tutoriales.

  BACKEND "tasks"   ->  vision.HandLandmarker / vision.PoseLandmarker
                        Disponible desde 0.10.x y UNICA en mediapipe 1.x.
                        Descarga bundles .task la primera vez.

El resto del proyecto (app.py, gestures, mapper, protocol) NO se entera de cual
se esta usando: la interfaz submit() / latest() es identica.

DIFERENCIAS REALES ENTRE BACKENDS (hay que conocerlas)
------------------------------------------------------
1. La API legacy es SINCRONA: `hands.process()` bloquea hasta terminar. Si se
   llamara desde el hilo de la GUI, la interfaz se congelaria cada vez que la
   inferencia se atrase, que es justo lo que prohibe el requisito de la
   semana 9. Por eso aqui corre en un HILO TRABAJADOR propio, con la misma
   politica que la camara: procesar siempre el frame mas nuevo y tirar los
   viejos. La API de Tasks ya trae eso resuelto con detect_async().

2. La Pose legacy es de UNA SOLA PERSONA. `mp.solutions.pose` no tiene
   parametro equivalente a num_poses. Si se necesita multi-persona real
   (requisito 1), hay que usar el backend "tasks" o aceptar la limitacion y
   documentarla. La deteccion de VARIAS MANOS si funciona en ambos backends,
   pero sin saber a quien pertenecen.

3. La legacy exige numpy<2. Es una restriccion del paquete, no negociable.
"""

from __future__ import annotations

import os
import sys
import threading
import urllib.request
from dataclasses import dataclass

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Falta mediapipe. Instala con:  pip install -r requirements.txt\n"
        "Recuerda que en Python 3.13 no hay wheel oficial: usa 3.11."
    ) from exc

from . import landmarks as LM
from .features import HandFeatures, extract


# ---------------------------------------------------------------------------
# Deteccion de backend disponible
# ---------------------------------------------------------------------------
def _has_legacy() -> bool:
    try:
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    except Exception:
        return False


def _has_tasks() -> bool:
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return True
    except Exception:
        return False


HAS_LEGACY = _has_legacy()
HAS_TASKS = _has_tasks()


def _mediapipe_major() -> int:
    try:
        return int(str(getattr(mp, "__version__", "0")).split(".")[0])
    except Exception:
        return 0


def _apple_silicon() -> bool:
    import platform
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def check_plataforma() -> None:
    """
    Guarda contra un bug conocido: mediapipe 1.0.x aborta el proceso en Mac con
    Apple Silicon.

        F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
            @ -[DrishtiMetalHelper initWithCalculatorContext:]
            @ mediapipe::api2::TensorsToDetectionsCalculator::Open()

    El calculador inicializa Metal de forma incondicional, asi que pasar
    delegate=CPU NO lo evita, y como es un abort() de C++ no se puede atrapar
    con try/except: se lleva al interprete completo. La rama 0.10.x si funciona.

    Se verifica ANTES de crear el landmarker para que el usuario reciba una
    explicacion en vez de un volcado de pila.
    """
    if not (_apple_silicon() and _mediapipe_major() >= 1):
        return
    if os.environ.get("SOFIA_FORZAR_MEDIAPIPE_1") == "1":
        print("[aviso] forzando mediapipe 1.x en Apple Silicon. "
              "Si el proceso aborta, ya sabes por que.")
        return

    raise SystemExit(
        "\n" + "=" * 68 + "\n"
        f"mediapipe {getattr(mp, '__version__', '?')} no funciona en Mac con "
        "Apple Silicon.\n\n"
        "La rama 1.0.x inicializa Metal incondicionalmente y aborta el proceso\n"
        "con 'Check failed: service_ Service is unavailable'. Es un abort() de\n"
        "C++, no una excepcion: no se puede manejar desde Python.\n\n"
        "Solucion (tambien te devuelve mp.solutions, la API legacy):\n\n"
        "    pip uninstall -y mediapipe opencv-python opencv-contrib-python\n"
        '    pip install "mediapipe==0.10.21" "numpy<2"\n'
        "    python diagnostico.py\n\n"
        "Nota: mediapipe 0.10.x instala opencv-contrib-python por su cuenta.\n"
        "No agregues opencv-python ademas: los dos paquetes entregan el modulo\n"
        "cv2 y se pisan entre si.\n\n"
        "Si de todas formas quieres intentarlo:  "
        "SOFIA_FORZAR_MEDIAPIPE_1=1 python app.py\n"
        + "=" * 68)


def resolve_backend(preferred: str = "auto") -> str:
    """
    Decide que backend usar.

    "auto" prefiere legacy cuando esta disponible, porque no requiere descargar
    modelos y arranca sin conexion a internet. Si no esta, cae a tasks.
    Se puede forzar con la variable de entorno SOFIA_BACKEND o con --backend.
    """
    preferred = os.environ.get("SOFIA_BACKEND", preferred).lower()

    if preferred == "legacy":
        if not HAS_LEGACY:
            raise RuntimeError(
                "Se pidio backend 'legacy' pero mp.solutions no existe en esta "
                f"instalacion (mediapipe {getattr(mp, '__version__', '?')}).\n"
                "Instala:  pip install mediapipe==0.10.21 \"numpy<2\""
            )
        return "legacy"

    if preferred == "tasks":
        if not HAS_TASKS:
            raise RuntimeError("Se pidio backend 'tasks' pero no esta disponible.")
        return "tasks"

    if HAS_LEGACY:
        return "legacy"
    if HAS_TASKS:
        return "tasks"
    raise RuntimeError("Ninguna API de mediapipe disponible. Revisa la instalacion.")


def backend_info() -> str:
    ver = getattr(mp, "__version__", "desconocida")
    return (f"mediapipe {ver} | legacy={'si' if HAS_LEGACY else 'no'} "
            f"| tasks={'si' if HAS_TASKS else 'no'}")


# ---------------------------------------------------------------------------
# Modelos .task (solo backend tasks)
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def ensure_model(name: str, models_dir: str = "models") -> str:
    """Descarga el bundle .task la primera vez y devuelve su ruta local."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.task")
    if not os.path.exists(path):
        print(f"[modelo] descargando {name} (solo la primera vez)...")
        urllib.request.urlretrieve(MODEL_URLS[name], path)
        print(f"[modelo] guardado en {path}")
    return path


def _to_array(landmark_list) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmark_list], dtype=float)


@dataclass
class FrameResult:
    hands: list[HandFeatures]
    poses: list[np.ndarray]
    timestamp_ms: int
    assignments: dict[int, int]


# ---------------------------------------------------------------------------
# Hilo trabajador para la API sincrona legacy
# ---------------------------------------------------------------------------
class _SyncWorker:
    """
    Convierte una API bloqueante en una asincrona.

    Guarda solo el frame mas reciente. Si llega uno nuevo mientras el anterior
    se procesa, el viejo se descarta: mas vale saltarse frames que acumular
    retardo, porque un gesto detectado 300 ms tarde ya no sirve para reaccionar.
    """

    def __init__(self, process_fn) -> None:
        self._process = process_fn
        self._pending: tuple[np.ndarray, int] | None = None
        self._result = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._event.wait(timeout=0.1):
                continue
            self._event.clear()
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                continue
            try:
                res = self._process(job[0])
            except Exception as exc:                      # no tumbar la app
                print(f"[mediapipe] error de inferencia: {exc}")
                continue
            with self._lock:
                self._result = (res, job[1])

    def submit(self, rgb: np.ndarray, timestamp_ms: int) -> None:
        with self._lock:
            self._pending = (rgb, timestamp_ms)
        self._event.set()

    def latest(self):
        with self._lock:
            return self._result

    def close(self) -> None:
        self._stop.set()
        self._event.set()
        self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------
class HandTracker:
    """Detector de manos. Misma interfaz en los dos backends."""

    def __init__(self, model_path: str | None = None, num_hands: int = 4,
                 det_conf: float = 0.5, pres_conf: float = 0.5,
                 track_conf: float = 0.5, image_size: tuple[int, int] = (640, 480),
                 backend: str = "auto", model_complexity: int = 1) -> None:
        self.backend = resolve_backend(backend)
        self.image_size = image_size
        self._lock = threading.Lock()
        self._latest = None
        self._latest_ts = 0

        if self.backend == "legacy":
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                model_complexity=model_complexity,   # 0 rapido, 1 preciso
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self._worker = _SyncWorker(self._hands.process)
        else:
            check_plataforma()
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = model_path or ensure_model("hand")
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=num_hands,
                min_hand_detection_confidence=det_conf,
                min_hand_presence_confidence=pres_conf,
                min_tracking_confidence=track_conf,
                result_callback=self._on_result,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._latest = result
            self._latest_ts = timestamp_ms

    def submit(self, rgb: np.ndarray, timestamp_ms: int) -> None:
        """Encola un frame. Retorna de inmediato en ambos backends."""
        if self.backend == "legacy":
            self._worker.submit(rgb, timestamp_ms)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._landmarker.detect_async(mp_image, timestamp_ms)

    # -- conversion a HandFeatures ------------------------------------------
    def _scale(self, pts: np.ndarray) -> np.ndarray:
        w, h = self.image_size
        pts = pts.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts[:, 2] *= w          # z viene en la misma escala que x
        return pts

    def _from_legacy(self, res, ts: int) -> list[HandFeatures]:
        if res is None or not res.multi_hand_landmarks:
            return []
        out = []
        for i, lms in enumerate(res.multi_hand_landmarks):
            pts = self._scale(_to_array(lms.landmark))

            world = pts
            wl = getattr(res, "multi_hand_world_landmarks", None)
            if wl:
                world = _to_array(wl[i].landmark)

            name, score = "Right", 1.0
            if res.multi_handedness:
                c = res.multi_handedness[i].classification[0]
                name, score = c.label, float(c.score)

            out.append(extract(pts, world, name, score, ts))
        return out

    def _from_tasks(self, res, ts: int) -> list[HandFeatures]:
        if res is None or not res.hand_landmarks:
            return []
        out = []
        for i, lms in enumerate(res.hand_landmarks):
            pts = self._scale(_to_array(lms))
            world = (_to_array(res.hand_world_landmarks[i])
                     if res.hand_world_landmarks else pts)
            cat = res.handedness[i][0] if res.handedness else None
            name = cat.category_name if cat else "Right"
            score = float(cat.score) if cat else 1.0
            out.append(extract(pts, world, name, score, ts))
        return out

    def latest(self) -> list[HandFeatures]:
        if self.backend == "legacy":
            item = self._worker.latest()
            if item is None:
                return []
            return self._from_legacy(item[0], item[1])

        with self._lock:
            res, ts = self._latest, self._latest_ts
        return self._from_tasks(res, ts)

    def close(self) -> None:
        if self.backend == "legacy":
            self._worker.close()
            self._hands.close()
        else:
            self._landmarker.close()


# ---------------------------------------------------------------------------
# PoseTracker
# ---------------------------------------------------------------------------
class PoseTracker:
    """
    Detector de cuerpos.

    OJO: con backend legacy es de UNA SOLA PERSONA (limitacion de
    mp.solutions.pose). Con backend tasks soporta num_poses > 1. Si el eje SC
    va a demostrar deteccion en varias personas, ese es el motivo para usar
    tasks aunque el resto del codigo corra igual en los dos.
    """

    def __init__(self, model_path: str | None = None, num_poses: int = 3,
                 det_conf: float = 0.5, image_size: tuple[int, int] = (640, 480),
                 backend: str = "auto", model_complexity: int = 1) -> None:
        self.backend = resolve_backend(backend)
        self.image_size = image_size
        self.max_poses = 1 if self.backend == "legacy" else num_poses
        self._lock = threading.Lock()
        self._latest = None

        if self.backend == "legacy":
            if num_poses > 1:
                print("[pose] backend legacy: solo se detectara 1 persona "
                      "(mp.solutions.pose no es multi-persona). "
                      "Usa --backend tasks si necesitas varias.")
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=det_conf,
                min_tracking_confidence=det_conf,
                enable_segmentation=False,
            )
            self._worker = _SyncWorker(self._pose.process)
        else:
            check_plataforma()
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = model_path or ensure_model("pose_lite")
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=num_poses,
                min_pose_detection_confidence=det_conf,
                min_pose_presence_confidence=det_conf,
                min_tracking_confidence=det_conf,
                output_segmentation_masks=False,
                result_callback=self._on_result,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._latest = result

    def submit(self, rgb: np.ndarray, timestamp_ms: int) -> None:
        if self.backend == "legacy":
            self._worker.submit(rgb, timestamp_ms)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._landmarker.detect_async(mp_image, timestamp_ms)

    def _scale(self, p: np.ndarray) -> np.ndarray:
        w, h = self.image_size
        p = p.copy()
        p[:, 0] *= w
        p[:, 1] *= h
        return p

    def latest(self) -> list[np.ndarray]:
        if self.backend == "legacy":
            item = self._worker.latest()
            if item is None or item[0] is None:
                return []
            res = item[0]
            if not res.pose_landmarks:
                return []
            return [self._scale(_to_array(res.pose_landmarks.landmark))]

        with self._lock:
            res = self._latest
        if res is None or not res.pose_landmarks:
            return []
        return [self._scale(_to_array(lms)) for lms in res.pose_landmarks]

    def close(self) -> None:
        if self.backend == "legacy":
            self._worker.close()
            self._pose.close()
        else:
            self._landmarker.close()


# ---------------------------------------------------------------------------
# Asociacion mano <-> persona
# ---------------------------------------------------------------------------
def associate_hands_to_poses(
    hands: list[HandFeatures],
    poses: list[np.ndarray],
    max_dist_px: float = 160.0,
) -> dict[int, int]:
    """
    Asigna cada mano a la persona mas cercana comparando la muneca de la mano
    con las munecas del esqueleto corporal (landmarks 15 y 16 de Pose).

    Se rechaza la asignacion si supera `max_dist_px`, para no pegarle una mano
    suelta a la persona equivocada cuando hay dos personas juntas.
    """
    if not poses:
        return {}

    out: dict[int, int] = {}
    for hi, h in enumerate(hands):
        wrist = h.points[LM.WRIST][:2]
        best, best_d = None, float("inf")
        for pi, p in enumerate(poses):
            for j in (LM.POSE_LEFT_WRIST, LM.POSE_RIGHT_WRIST):
                d = float(np.linalg.norm(p[j][:2] - wrist))
                if d < best_d:
                    best, best_d = pi, d
        if best is not None and best_d <= max_dist_px:
            out[hi] = best
    return out


def hand_key(h: HandFeatures, person: int | None) -> str:
    """
    Identificador estable de una mano a traves de los frames. Se usa como clave
    del GestureEngine: sin esto, el estado de las maquinas dinamicas se
    mezclaria entre manos distintas.
    """
    return f"p{person if person is not None else 'x'}_{h.handedness}"