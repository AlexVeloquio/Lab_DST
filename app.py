"""
sOFIa - Eje Tecnico SC
Aplicacion principal: camara -> MediaPipe -> gestos -> matriz 16 -> ESP32.

Uso:
    python app.py                          # solo vision, sin hardware
    python app.py --host 192.168.4.1       # enviando a la ESP32 Maestra
    python app.py --mapeo dedos --pose     # mapeo semantico + multi-persona

Teclas:
    q  salir           m  alternar mapeo espejo/dedos
    i  ids de landmarks  p  activar/desactivar deteccion de cuerpo
    espacio  pausar/reanudar procesamiento
    s  guardar captura   r  reiniciar filtros

ARQUITECTURA DE HILOS
---------------------
  hilo captura   : lee de la camara sin parar y guarda solo el frame mas nuevo
                   (descartar frames viejos es preferible a acumular retardo)
  hilo MediaPipe : interno de la libreria, entrega resultados por callback
  hilo principal : dibuja, decide y envia

De ahi que la GUI nunca se congele aunque la inferencia baje de velocidad:
es exactamente el requisito de la semana 9 sobre no bloquear la interfaz.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from core import protocol
from core.gestures import GestureEngine
from core.mapper import GRID, ModuleMapper, to_bytes
from core.smoothing import LandmarkSmoother, SlewLimiter
from core.tracker import (
    HandTracker, PoseTracker, associate_hands_to_poses, backend_info, hand_key,
)
from ui import overlay


def _capture_backends() -> list[int]:
    """
    Backend de captura segun el sistema operativo.

    CAP_DSHOW es EXCLUSIVO de Windows (DirectShow). En macOS o Linux pedirlo
    hace que VideoCapture falle o tarde en abrir. En macOS lo correcto es
    AVFoundation y en Linux V4L2. Siempre se deja CAP_ANY como ultimo recurso.
    """
    if sys.platform.startswith("win"):
        nombres = ["CAP_DSHOW", "CAP_MSMF"]
    elif sys.platform == "darwin":
        nombres = ["CAP_AVFOUNDATION"]
    else:
        nombres = ["CAP_V4L2"]
    backends = [getattr(cv2, n) for n in nombres if hasattr(cv2, n)]
    backends.append(cv2.CAP_ANY)
    return backends


class CameraThread:
    """
    Captura desacoplada. Politica: siempre el frame mas reciente.

    index puede ser un entero o "auto". Con "auto" se prueban los indices 0 a 3
    y se usa el primero que entregue un frame de verdad: en varias Mac la camara
    integrada no es el indice 0, y abrir el indice equivocado da una ventana
    negra sin ningun mensaje de error.
    """

    def __init__(self, index=0, width: int = 640, height: int = 480):
        self.index, self.width, self.height = index, width, height
        self.frame: np.ndarray | None = None
        self.ok = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "CameraThread":
        self._t.start()
        return self

    def _try_open(self, idx: int):
        for backend in _capture_backends():
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
            cap.release()
        return None

    def _open(self) -> bool:
        indices = range(4) if self.index == "auto" else [int(self.index)]
        for idx in indices:
            cap = self._try_open(idx)
            if cap is None:
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # no soportado en todos
            except Exception:
                pass
            if self.index == "auto":
                print(f"[camara] usando el indice {idx}")
            self.index = idx
            self._cap = cap
            return True
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None and not self._open():
                self.ok = False
                time.sleep(1.0)          # reintento de reconexion de camara
                continue
            ret, frame = self._cap.read()
            if not ret:
                self.ok = False
                self._cap.release()
                self._cap = None
                continue
            with self._lock:
                self.frame = frame
                self.ok = True

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="sOFIa - Eje Tecnico SC")
    ap.add_argument("--cam", default="auto",
                    help="indice de camara, o 'auto' para buscarla sola")
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--manos", type=int, default=2)
    ap.add_argument("--host", default=None, help="IP de la ESP32 Maestra")
    ap.add_argument("--puerto", type=int, default=4210)
    ap.add_argument("--mapeo", choices=ModuleMapper.STRATEGIES, default="espejo")
    ap.add_argument("--pose", action="store_true", help="detectar cuerpos")
    ap.add_argument("--hz", type=float, default=25.0, help="tasa de envio MATRIX16")
    ap.add_argument("--metricas", default=None, help="archivo CSV de metricas")
    ap.add_argument("--backend", choices=["auto", "legacy", "tasks"], default="auto",
                    help="legacy = mp.solutions | tasks = HandLandmarker")
    ap.add_argument("--complejidad", type=int, choices=[0, 1], default=1,
                    help="0 modelo rapido, 1 modelo preciso (solo backend legacy)")
    args = ap.parse_args()

    print(f"[entorno] {backend_info()}")

    cam_idx = args.cam if args.cam == "auto" else int(args.cam)
    size = (args.ancho, args.alto)
    cam = CameraThread(cam_idx, args.ancho, args.alto).start()

    hands_tracker = HandTracker(num_hands=args.manos, image_size=size,
                                backend=args.backend,
                                model_complexity=args.complejidad)
    pose_tracker = (PoseTracker(image_size=size, backend=args.backend,
                                model_complexity=args.complejidad)
                    if args.pose else None)
    print(f"[entorno] backend en uso: {hands_tracker.backend}")

    engine = GestureEngine()
    mapper = ModuleMapper(args.mapeo)
    slew = SlewLimiter(size=GRID * GRID, max_rate=420.0, deadband=4.0)
    smoothers: dict[str, LandmarkSmoother] = {}

    link = protocol.SofiaLink(args.host, args.puerto) if args.host else None
    if link:
        print(f"[enlace] handshake con {args.host}:{args.puerto} ...",
              "ok" if link.hello() else "sin respuesta (se reintenta solo)")

    csv_file = csv_writer = None
    if args.metricas:
        csv_file = open(args.metricas, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t_ms", "fps", "fps_p5", "manos", "gesto",
                             "confianza", "proceso_ms"])

    t_start = time.perf_counter()
    fps_hist: deque[float] = deque(maxlen=60)
    t_frame_prev: float | None = None
    last_send = 0.0
    send_period = 1.0 / max(args.hz, 1.0)
    show_ids = paused = False
    last_gesture = "NINGUNO"
    frame_id = 0

    print("Listo. q salir | m mapeo | i ids | espacio pausa | s captura")
    print("Si la ventana no aparece, revisa que ninguna otra app use la camara.")
    if sys.platform == "darwin":
        print("macOS: la terminal necesita permiso en Ajustes > Privacidad y "
              "seguridad > Camara.")

    try:
        while True:
            # PERIODO REAL entre cuadros, no el tiempo de una parte del ciclo.
            # Midiendo solo hasta antes de imshow/waitKey se excluian las dos
            # operaciones mas caras y el contador reportaba cientos de FPS.
            loop_t0 = time.perf_counter()
            if t_frame_prev is not None:
                periodo = loop_t0 - t_frame_prev
                if periodo > 1e-6:
                    fps_hist.append(1.0 / periodo)
            t_frame_prev = loop_t0

            frame = cam.read()

            if frame is None:
                canvas = np.full((args.alto, args.ancho, 3), 30, np.uint8)
                overlay.draw_banner(canvas, "Camara no disponible. Reintentando...")
                cv2.imshow("sOFIa - Eje SC", canvas)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)          # espejo: mas natural para el usuario
            now = time.perf_counter()
            ts_ms = int((now - t_start) * 1000)  # monotono y creciente: obligatorio

            if not paused:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands_tracker.submit(rgb, ts_ms)
                if pose_tracker:
                    pose_tracker.submit(rgb, ts_ms + 1)

            hands = hands_tracker.latest()
            poses = pose_tracker.latest() if pose_tracker else []
            assign = associate_hands_to_poses(hands, poses)

            # --- suavizado de landmarks por mano ---------------------------
            seen: set[str] = set()
            for hi, h in enumerate(hands):
                key = hand_key(h, assign.get(hi))
                seen.add(key)
                sm = smoothers.setdefault(key, LandmarkSmoother())
                h.points = sm(h.points, now)

            for stale in set(smoothers) - seen:
                smoothers.pop(stale, None)
                engine.drop(stale)

            # --- gestos ----------------------------------------------------
            event = None
            for hi, h in enumerate(hands):
                ev = engine.update(hand_key(h, assign.get(hi)), h)
                if ev is not None:
                    event = ev
            if event:
                last_gesture = event.name
                print(f"[gesto] {event.name} conf={event.confidence:.2f} "
                      f"{event.hand} {event.payload or ''}")
                if link:
                    # direccion viaja como int8: -1 se envia como 0xFF
                    extra = int((event.payload or {}).get("direccion", 0)) & 0xFF
                    ok = link.send_gesture(event.name, event.confidence, extra)
                    if not ok:
                        print("[enlace] sin ACK del gesto")

            # --- matriz 16 -------------------------------------------------
            target = mapper(hands)
            matrix = slew(target, now)

            if link:
                link.tick()
                if now - last_send >= send_period:
                    last_send = now
                    link.send_matrix(to_bytes(matrix))

            # --- dibujo ----------------------------------------------------
            for p in poses:
                overlay.draw_pose(frame, p)
            for h in hands:
                overlay.draw_hand(frame, h, show_ids)
            if hands:
                overlay.draw_finger_bars(frame, hands[0])
                overlay.draw_palm_gauge(frame, hands[0])
            overlay.draw_matrix(frame, matrix)

            dt = time.perf_counter() - loop_t0          # costo de procesamiento
            fps = float(np.mean(fps_hist)) if fps_hist else 0.0
            fps_p5 = (float(np.percentile(fps_hist, 5))
                      if len(fps_hist) > 5 else fps)

            overlay.draw_hud(frame, fps, len(hands), len(poses), last_gesture,
                             bool(link and link.connected), mapper.strategy)
            if paused:
                overlay.draw_banner(frame, "PAUSADO - espacio para reanudar",
                                    (40, 120, 190))

            cv2.imshow("sOFIa - Eje SC", frame)

            if csv_writer:
                csv_writer.writerow([
                    ts_ms, f"{fps:.2f}", f"{fps_p5:.2f}", len(hands),
                    event.name if event else "",
                    f"{event.confidence:.3f}" if event else "",
                    f"{dt * 1000:.1f}",
                ])

            # --- teclado ---------------------------------------------------
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("m"):
                other = "dedos" if mapper.strategy == "espejo" else "espejo"
                mapper = ModuleMapper(other)
            elif k == ord("i"):
                show_ids = not show_ids
            elif k == ord(" "):
                paused = not paused
            elif k == ord("r"):
                smoothers.clear()
            elif k == ord("s"):
                frame_id += 1
                name = f"captura_{frame_id:03d}.png"
                cv2.imwrite(name, frame)
                print(f"[captura] {name}")

    except KeyboardInterrupt:
        pass
    finally:
        if link:
            link.stop()                     # paro seguro de los servos
            link.close()
        if csv_file:
            csv_file.close()
        hands_tracker.close()
        if pose_tracker:
            pose_tracker.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("Cerrado limpiamente.")


if __name__ == "__main__":
    main()
