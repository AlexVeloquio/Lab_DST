"""
Sesion guiada de evaluacion. Genera el registro crudo de las pruebas
experimentales (punto 3 del entregable).

    # Bloque A: robustez entre usuarios
    python evaluacion.py --usuario U1 --iluminacion laboratorio --distancia 2.0 \
                         --repeticiones 20

    # Bloque B: robustez ante condiciones de captura
    python evaluacion.py --usuario U1 --iluminacion contraluz --distancia 3.5 \
                         --repeticiones 10

    # Ventana de reposo: mide falsos positivos
    python evaluacion.py --usuario U1 --reposo 3

Como funciona
-------------
El script anuncia una postura objetivo en ORDEN ALEATORIO, da 2 s de
preparacion y abre una ventana de 4 s. Lo que el motor detecte dentro de esa
ventana se anota junto con la postura que se pidio y el instante en que llego.

El orden aleatorio no es un detalle: repetir veinte veces seguidas la misma
postura infla la tasa de acierto, porque el usuario se calibra contra el
detector, y ademas impide medir confusiones entre posturas, que es lo que
llena la matriz de confusion.

Salida: un CSV por sesion en resultados/. Se analiza con metricas.py.

Teclas durante la sesion:  q salir   s saltar el intento actual
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime

import cv2
import numpy as np

from app import CameraThread
from core.gestures import GestureEngine
from core.smoothing import LandmarkSmoother
from core.tracker import HandTracker, backend_info, hand_key
from ui import overlay

POSTURAS = ["PUNO", "PAZ", "SALUDO", "PULSO", "GIRO"]

INSTRUCCIONES = {
    "PUNO": "Cierra el puno por completo y manten quieto",
    "PAZ": "Indice y medio arriba, anular y menique abajo, quieto",
    "SALUDO": "Saluda moviendo la mano abierta de lado a lado",
    "PULSO": "Abre, cierra y vuelve a abrir rapido",
    "GIRO": "Voltea la mano de palma a dorso (o al reves)",
}

PREP_MS = 2000        # aviso antes de abrir la ventana
VENTANA_MS = 4000     # tiempo para ejecutar la postura
DESCANSO_MS = 1500    # mayor que el refractario de 900 ms


def _texto(img, txt, y, escala=0.7, color=(240, 240, 245), grosor=2):
    (w, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                color, grosor, cv2.LINE_AA)


def _panel(img, alto=110):
    cv2.rectangle(img, (0, 0), (img.shape[1], alto), (22, 22, 26), -1)


class Sesion:
    """Estado de una sesion de evaluacion y escritura del registro."""

    def __init__(self, args) -> None:
        self.args = args
        os.makedirs("resultados", exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        modo = "reposo" if args.reposo else "intentos"
        self.ruta = os.path.join(
            "resultados",
            f"{modo}_{args.usuario}_{args.iluminacion}_{args.distancia}m_{marca}.csv")
        self.f = open(self.ruta, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "usuario", "iluminacion", "distancia_m", "modo", "intento",
            "esperada", "detectada", "latencia_ms", "confianza",
            "fps_medio", "fps_p5",
        ])
        self.filas = 0

    def registrar(self, **kw) -> None:
        self.w.writerow([
            self.args.usuario, self.args.iluminacion, self.args.distancia,
            kw.get("modo", "intento"), kw.get("intento", ""),
            kw.get("esperada", ""), kw.get("detectada", "NINGUNA"),
            kw.get("latencia_ms", ""), kw.get("confianza", ""),
            f"{kw.get('fps_medio', 0):.2f}", f"{kw.get('fps_p5', 0):.2f}",
        ])
        self.f.flush()
        self.filas += 1

    def cerrar(self) -> str:
        self.f.close()
        return self.ruta


def correr(args) -> str:
    print(f"[entorno] {backend_info()}")
    size = (args.ancho, args.alto)
    cam_idx = args.cam if args.cam == "auto" else int(args.cam)
    cam = CameraThread(cam_idx, args.ancho, args.alto).start()
    tracker = HandTracker(num_hands=1, image_size=size, backend=args.backend)
    engine = GestureEngine()
    smoothers: dict[str, LandmarkSmoother] = {}
    sesion = Sesion(args)

    # Plan de la sesion: orden aleatorio, no por bloques de la misma postura
    if args.reposo:
        plan = []
        total_ms = args.reposo * 60_000
    else:
        plan = POSTURAS * args.repeticiones
        random.shuffle(plan)
        total_ms = 0

    t_inicio = time.perf_counter()
    fps_hist: list[float] = []
    t_frame_prev: float | None = None
    idx = 0
    fase = "prep"
    t_fase = 0.0
    detectado: tuple[str, float, float] | None = None
    resumen: list[tuple[str, str]] = []

    print(f"\nSesion: {len(plan) if plan else args.reposo} "
          f"{'intentos' if plan else 'minutos de reposo'}")
    print("q para salir, s para saltar un intento\n")

    try:
        while True:
            # Periodo real entre cuadros (ver la nota en app.py)
            t_loop = time.perf_counter()
            if t_frame_prev is not None and t_loop - t_frame_prev > 1e-6:
                fps_hist.append(1.0 / (t_loop - t_frame_prev))
            t_frame_prev = t_loop

            frame = cam.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            ahora = time.perf_counter()
            ts_ms = int((ahora - t_inicio) * 1000)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.submit(rgb, ts_ms)
            manos = tracker.latest()

            evento = None
            for h in manos:
                k = hand_key(h, None)
                sm = smoothers.setdefault(k, LandmarkSmoother())
                h.points = sm(h.points, ahora)
                ev = engine.update(k, h)
                if ev is not None:
                    evento = ev

            for h in manos:
                overlay.draw_hand(frame, h)
            if manos:
                overlay.draw_palm_gauge(frame, manos[0])

            fps_hist = fps_hist[-300:]
            fps_medio = float(np.mean(fps_hist)) if fps_hist else 0.0
            fps_p5 = (float(np.percentile(fps_hist, 5))
                      if len(fps_hist) > 5 else fps_medio)

            # ----------------------------------------------------------------
            # MODO REPOSO: toda deteccion es un falso positivo
            # ----------------------------------------------------------------
            if args.reposo:
                transcurrido = ts_ms
                if evento is not None:
                    sesion.registrar(modo="reposo", esperada="NINGUNA",
                                     detectada=evento.name,
                                     latencia_ms=transcurrido,
                                     confianza=f"{evento.confidence:.3f}",
                                     fps_medio=fps_medio, fps_p5=fps_p5)
                    print(f"  falso positivo: {evento.name} "
                          f"a los {transcurrido/1000:.1f} s")

                _panel(frame)
                _texto(frame, "REPOSO - no ejecutes ninguna postura", 40, 0.7)
                restante = max(0, (total_ms - transcurrido) / 1000)
                _texto(frame, f"quedan {restante:5.1f} s   "
                              f"falsos positivos: {sesion.filas}", 80, 0.6,
                       (150, 150, 160))
                if transcurrido >= total_ms:
                    break

            # ----------------------------------------------------------------
            # MODO INTENTOS
            # ----------------------------------------------------------------
            else:
                if idx >= len(plan):
                    break
                objetivo = plan[idx]
                elapsed = (ahora - t_fase) * 1000 if t_fase else 1e9

                if fase == "prep":
                    if t_fase == 0.0:
                        t_fase = ahora
                        elapsed = 0
                    _panel(frame)
                    _texto(frame, f"{idx+1}/{len(plan)}   PREPARATE", 36, 0.7,
                           (150, 150, 160))
                    _texto(frame, objetivo, 72, 1.0, (70, 200, 255))
                    _texto(frame, INSTRUCCIONES[objetivo], 100, 0.5,
                           (190, 190, 200))
                    if elapsed >= PREP_MS:
                        fase, t_fase = "ventana", ahora
                        detectado = None
                        engine.drop(hand_key(manos[0], None)) if manos else None

                elif fase == "ventana":
                    if evento is not None and detectado is None:
                        detectado = (evento.name, evento.confidence, elapsed)
                    _panel(frame)
                    _texto(frame, f"{idx+1}/{len(plan)}   EJECUTA", 36, 0.7,
                           (90, 220, 120))
                    _texto(frame, objetivo, 72, 1.0, (70, 200, 255))
                    barra = int((1 - elapsed / VENTANA_MS) * frame.shape[1])
                    cv2.rectangle(frame, (0, 104), (max(barra, 0), 110),
                                  (90, 220, 120), -1)
                    if detectado or elapsed >= VENTANA_MS:
                        nombre = detectado[0] if detectado else "NINGUNA"
                        sesion.registrar(
                            intento=idx + 1, esperada=objetivo,
                            detectada=nombre,
                            latencia_ms=f"{detectado[2]:.0f}" if detectado else "",
                            confianza=f"{detectado[1]:.3f}" if detectado else "",
                            fps_medio=fps_medio, fps_p5=fps_p5)
                        resumen.append((objetivo, nombre))
                        ok = "ok " if nombre == objetivo else "   "
                        print(f"  {ok}{idx+1:3d}/{len(plan)}  pedida {objetivo:7s}"
                              f"  detectada {nombre}")
                        fase, t_fase = "descanso", ahora

                else:  # descanso
                    _panel(frame)
                    ultima = resumen[-1] if resumen else ("", "")
                    acierto = ultima[0] == ultima[1]
                    _texto(frame, "CORRECTO" if acierto else f"-> {ultima[1]}",
                           60, 0.9,
                           (90, 220, 120) if acierto else (90, 120, 230))
                    if elapsed >= DESCANSO_MS:
                        idx += 1
                        fase, t_fase = "prep", 0.0

            cv2.putText(frame, f"FPS {fps_medio:4.1f} (p5 {fps_p5:4.1f})",
                        (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (140, 140, 150), 1, cv2.LINE_AA)
            cv2.imshow("sOFIa - evaluacion", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s") and not args.reposo:
                idx += 1
                fase, t_fase = "prep", 0.0

    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        cam.stop()
        cv2.destroyAllWindows()

    ruta = sesion.cerrar()
    if resumen:
        aciertos = sum(1 for e, d in resumen if e == d)
        print(f"\nAciertos: {aciertos}/{len(resumen)} "
              f"({100*aciertos/len(resumen):.1f} %)")
    print(f"Registro guardado en: {ruta}")
    print(f"Analizalo con:  python metricas.py {ruta}")
    return ruta


def main() -> None:
    ap = argparse.ArgumentParser(description="Sesion guiada de evaluacion")
    ap.add_argument("--usuario", required=True, help="identificador, ej. U1")
    ap.add_argument("--iluminacion", default="laboratorio",
                    choices=["laboratorio", "contraluz", "calida"])
    ap.add_argument("--distancia", default="2.0", help="metros, ej. 2.0")
    ap.add_argument("--repeticiones", type=int, default=20,
                    help="repeticiones POR POSTURA")
    ap.add_argument("--reposo", type=int, default=0,
                    help="minutos de ventana de reposo (mide falsos positivos)")
    ap.add_argument("--cam", default="auto")
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--backend", choices=["auto", "legacy", "tasks"],
                    default="auto")
    ap.add_argument("--semilla", type=int, default=None,
                    help="fija el orden aleatorio, para reproducir una sesion")
    args = ap.parse_args()

    if args.semilla is not None:
        random.seed(args.semilla)
    correr(args)


if __name__ == "__main__":
    main()
