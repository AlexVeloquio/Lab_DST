"""
sOFIa - Eje Tecnico SC
Interfaz grafica: video en vivo, resultado de deteccion, control del
procesamiento y enlace con la ESP32 Maestra.

    python gui.py                          # solo vision
    python gui.py --host 192.168.4.1       # con la ESP32 real
    python gui.py --host 127.0.0.1         # contra esp32_simulador.py

POR QUE TKINTER
---------------
Viene en la biblioteca estandar de Python: cero dependencias nuevas, y el
entregable corre en cualquier maquina del laboratorio con solo clonar el
repositorio. Para esta GUI, que son controles, etiquetas y un lienzo de video,
alcanza de sobra.

Las alternativas se descartaron por costo, no por capacidad. PyQt es mas
potente pero agrega ~100 MB y licencia GPL o comercial. Un front web con Flask
anadiria un servidor y un navegador en medio del camino de video, justo donde
importa la latencia. La ventana nativa de OpenCV no es una GUI: no tiene
botones ni campos de texto reales.

Limitacion asumida: Tk solo puede tocarse desde el hilo principal. De ahi la
arquitectura de abajo.

ARQUITECTURA DE HILOS
---------------------
    Hilo camara        lee sin parar, conserva solo el cuadro mas nuevo
    Hilo procesamiento MediaPipe, gestos, mapeo; publica en una cola de 1
    Hilo enlace        ping y handshake con la ESP32, para no congelar nada
    Hilo principal     Tk: dibuja y atiende botones. NUNCA bloquea

El hilo principal no espera a nadie: cada 20 ms toma lo ultimo que haya en la
cola con root.after(). Si el procesamiento se atrasa, la interfaz sigue viva y
el video simplemente repite el cuadro anterior. Descartar cuadros viejos es
preferible a acumular retardo: un gesto detectado 300 ms tarde ya no sirve
para reaccionar.

MAQUINA DE ESTADOS DE LOS CONTROLES
-----------------------------------
    DETENIDO  --Iniciar-->  EJECUTANDO  --Detener-->  PAUSADO
                                ^                        |
                                +------Reanudar----------+

En PAUSADO el video sigue vivo pero el hilo de procesamiento se queda en una
espera pasiva sobre un threading.Event: no consume CPU y no es un bucle que
gira en vacio. Al reanudar se limpian los detectores dinamicos, porque sus
ventanas temporales quedaron con datos de antes de la pausa.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk

import cv2
import numpy as np

from core import protocol
from core.gestures import GestureEngine
from core.mapper import GRID, ModuleMapper, to_bytes
from core.smoothing import LandmarkSmoother, SlewLimiter
from core.tracker import HandTracker, backend_info, hand_key
from ui import overlay

# --- paleta, alineada con el HUD de OpenCV -------------------------------
BG = "#17171b"
PANEL = "#1f1f25"
CARD = "#26262e"
FG = "#ebebf0"
MUTED = "#8a8a96"
ACCENT = "#46c8ff"
OK = "#5adc78"
WARN = "#ffb648"
BAD = "#ff6b6b"

ESTADOS = ("DETENIDO", "EJECUTANDO", "PAUSADO")


@dataclass
class Cuadro:
    """Lo que el hilo de procesamiento publica para que la GUI lo pinte."""
    imagen: np.ndarray
    fps: float = 0.0
    fps_p5: float = 0.0
    manos: int = 0
    extensiones: dict = field(default_factory=dict)
    orientacion: str = "-"
    palm_normal: float = 0.0
    matriz: np.ndarray = field(default_factory=lambda: np.zeros(16))
    evento: object = None


# =========================================================================
# Hilos de trabajo
# =========================================================================
class HiloCamara(threading.Thread):
    """Captura desacoplada. Politica: siempre el cuadro mas reciente."""

    def __init__(self, indice="auto", ancho=640, alto=480):
        super().__init__(daemon=True)
        # Ojo: el evento NO puede llamarse _stop. threading.Thread usa ese
        # nombre para un metodo interno y sobrescribirlo rompe join().
        self.indice, self.ancho, self.alto = indice, ancho, alto
        self.cuadro: np.ndarray | None = None
        self.ok = False
        self._lock = threading.Lock()
        self._fin = threading.Event()
        self._cap = None

    @staticmethod
    def _backends():
        if sys.platform.startswith("win"):
            nombres = ["CAP_DSHOW", "CAP_MSMF"]
        elif sys.platform == "darwin":
            nombres = ["CAP_AVFOUNDATION"]
        else:
            nombres = ["CAP_V4L2"]
        return [getattr(cv2, n) for n in nombres if hasattr(cv2, n)] + [cv2.CAP_ANY]

    def _abrir(self) -> bool:
        indices = range(4) if self.indice == "auto" else [int(self.indice)]
        for idx in indices:
            for backend in self._backends():
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception:
                    continue
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        self.indice = idx
                        self._cap = cap
                        return True
                cap.release()
        return False

    def run(self) -> None:
        while not self._fin.is_set():
            if self._cap is None and not self._abrir():
                self.ok = False
                time.sleep(1.0)
                continue
            ret, frame = self._cap.read()
            if not ret:
                self.ok = False
                self._cap.release()
                self._cap = None
                continue
            with self._lock:
                self.cuadro = frame
                self.ok = True

    def leer(self):
        with self._lock:
            return None if self.cuadro is None else self.cuadro.copy()

    def detener(self) -> None:
        self._fin.set()
        self.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()


class HiloProcesamiento(threading.Thread):
    """
    MediaPipe, gestos y mapeo. Publica en una cola de tamano 1.

    `activo` es un threading.Event: cuando esta en claro, el hilo se bloquea
    en wait() y no consume CPU. Eso es lo que hace que el boton Detener tenga
    un efecto real sobre el hilo y no solo sobre lo que se dibuja.
    """

    def __init__(self, camara, args, salida: queue.Queue,
                 eventos: queue.Queue, log) -> None:
        super().__init__(daemon=True)
        self.cam = camara
        self.args = args
        self.salida = salida
        # Los eventos NO viajan en la cola de video. Esa cola es de tamano 1 y
        # descarta el cuadro anterior en cada publicacion: como el hilo
        # produce mucho mas rapido de lo que la GUI consume, el cuadro que
        # llevaba la postura se tiraba antes de que nadie lo viera. Una cola
        # aparte y sin limite garantiza que ninguna postura se pierda.
        self.eventos = eventos
        self.log = log
        self.activo = threading.Event()
        self._fin = threading.Event()
        self._reset = threading.Event()
        self.enlace: protocol.SofiaLink | None = None
        self.enviar_matriz = True
        self.estrategia = "espejo"
        self.error: str | None = None

    def reiniciar_detectores(self) -> None:
        self._reset.set()

    def detener(self) -> None:
        self._fin.set()
        self.activo.set()          # desbloquear si estaba esperando
        self.join(timeout=2.0)

    def run(self) -> None:
        try:
            tracker = HandTracker(num_hands=self.args.manos,
                                  image_size=(self.args.ancho, self.args.alto),
                                  backend=self.args.backend)
            self.log(f"Backend de MediaPipe: {tracker.backend}")
        except Exception as exc:
            self.error = str(exc)
            self.log(f"ERROR al iniciar MediaPipe: {exc}", nivel="error")
            return

        motor = GestureEngine()
        mapper = ModuleMapper(self.estrategia)
        slew = SlewLimiter(size=GRID * GRID)
        suavizadores: dict[str, LandmarkSmoother] = {}

        t0 = time.perf_counter()
        fps_hist: list[float] = []
        t_prev = None
        ultimo_envio = 0.0
        ultimo_ts = -1          # timestamp del ultimo resultado ya procesado
        periodo_envio = 1.0 / max(self.args.hz, 1.0)

        while not self._fin.is_set():
            self.activo.wait()                  # espera pasiva si esta pausado
            if self._fin.is_set():
                break

            if self._reset.is_set():
                self._reset.clear()
                motor = GestureEngine()
                suavizadores.clear()
                fps_hist.clear()
                t_prev = None

            if mapper.strategy != self.estrategia:
                mapper = ModuleMapper(self.estrategia)

            ahora = time.perf_counter()
            if t_prev is not None and ahora - t_prev > 1e-6:
                fps_hist.append(1.0 / (ahora - t_prev))
                fps_hist = fps_hist[-60:]
            t_prev = ahora

            frame = self.cam.leer()
            if frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            ts_ms = int((ahora - t0) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.submit(rgb, ts_ms)
            manos = tracker.latest()

            # MediaPipe entrega resultados por callback: latest() devuelve el
            # MISMO resultado hasta que llega uno nuevo. Sin este filtro, los
            # detectores reciben cientos de copias del mismo cuadro por
            # segundo, con el mismo timestamp, y las ventanas temporales se
            # llenan de duplicados.
            nuevo = bool(manos) and manos[0].timestamp_ms != ultimo_ts
            if manos:
                ultimo_ts = manos[0].timestamp_ms

            vistos = set()
            evento = None
            for h in manos if nuevo else []:
                k = hand_key(h, None)
                vistos.add(k)
                sm = suavizadores.setdefault(k, LandmarkSmoother())
                h.points = sm(h.points, ahora)
                ev = motor.update(k, h)
                if ev is not None:
                    evento = ev
            if nuevo:
                for viejo in set(suavizadores) - vistos:
                    suavizadores.pop(viejo, None)
                    motor.drop(viejo)

            matriz = slew(mapper(manos), ahora)

            if evento is not None:
                self.eventos.put(evento)     # cola sin limite: no se pierde

            if evento is not None and self.enlace is not None:
                extra = int((evento.payload or {}).get("direccion", 0)) & 0xFF
                ok = self.enlace.send_gesture(evento.name, evento.confidence, extra)
                self.log(f"{evento.name} enviado a la ESP32"
                         f"{'' if ok else ' (sin ACK)'}",
                         nivel="ok" if ok else "warn")

            if self.enlace is not None and self.enviar_matriz:
                self.enlace.tick()
                if ahora - ultimo_envio >= periodo_envio:
                    ultimo_envio = ahora
                    self.enlace.send_matrix(to_bytes(matriz))

            for h in manos:
                overlay.draw_hand(frame, h)

            c = Cuadro(
                imagen=frame,
                fps=float(np.mean(fps_hist)) if fps_hist else 0.0,
                fps_p5=float(np.percentile(fps_hist, 5)) if len(fps_hist) > 5 else 0.0,
                manos=len(manos),
                extensiones=dict(manos[0].extensions) if manos else {},
                orientacion=manos[0].orientacion if manos else "-",
                palm_normal=manos[0].palm_normal if manos else 0.0,
                matriz=matriz,
                evento=evento,
            )
            try:
                self.salida.get_nowait()        # descartar el cuadro anterior
            except queue.Empty:
                pass
            self.salida.put_nowait(c)

            # Techo de ritmo. Sin esto el bucle gira tan rapido como la CPU
            # permita, reencolando el mismo cuadro miles de veces por segundo:
            # consume un nucleo entero y falsea el contador de FPS.
            sobra = (1.0 / 60.0) - (time.perf_counter() - ahora)
            if sobra > 0:
                time.sleep(sobra)

        tracker.close()


# =========================================================================
# Interfaz
# =========================================================================
class App:
    def __init__(self, root: tk.Tk, args) -> None:
        self.root = root
        self.args = args
        self.estado = "DETENIDO"
        self.cola: queue.Queue[Cuadro] = queue.Queue(maxsize=1)
        # Cola de trabajo hacia el hilo de Tk. Ningun hilo toca widgets.
        self.cola_ui: queue.Queue[tuple] = queue.Queue()
        self.eventos: queue.Queue = queue.Queue()
        self.enlace: protocol.SofiaLink | None = None
        self.ultimo_cuadro: Cuadro | None = None
        self._img_ref = None
        self.historial: list[str] = []

        root.title("sOFIa · Eje Técnico SC · Interfaz de control")
        root.configure(bg=BG)
        root.geometry("1180x720")
        root.minsize(1040, 660)
        root.protocol("WM_DELETE_WINDOW", self.cerrar)

        self._estilos()
        self._construir()

        self.camara = HiloCamara(args.cam, args.ancho, args.alto)
        self.camara.start()
        self.proc = HiloProcesamiento(self.camara, args, self.cola,
                                      self.eventos, self.log)
        self.proc.start()

        if args.transporte == "ble":
            self.root.after(50, self._cambiar_transporte)
        else:
            self.btn_buscar.state(["disabled"])
        self.log(f"Entorno: {backend_info()}")
        self.log("Listo. Pulsa Iniciar para comenzar la detección.")
        if args.host:
            self.ip_var.set(args.host)
            self.puerto_var.set(str(args.puerto))
            self.root.after(400, self.conectar)

        self.root.after(30, self.refrescar)

    # -- construccion -----------------------------------------------------
    def _estilos(self) -> None:
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TFrame", background=PANEL)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=PANEL, foreground=FG,
                     font=("Helvetica", 10))
        st.configure("Card.TLabel", background=CARD, foreground=FG,
                     font=("Helvetica", 10))
        st.configure("Muted.TLabel", background=CARD, foreground=MUTED,
                     font=("Helvetica", 9))
        st.configure("Titulo.TLabel", background=CARD, foreground=MUTED,
                     font=("Helvetica", 9, "bold"))
        st.configure("Gesto.TLabel", background=CARD, foreground=ACCENT,
                     font=("Helvetica", 22, "bold"))
        st.configure("TButton", font=("Helvetica", 10), padding=6)
        st.configure("TEntry", fieldbackground="#30303a", foreground=FG)

    def _tarjeta(self, padre, titulo: str) -> ttk.Frame:
        cont = ttk.Frame(padre, style="Card.TFrame", padding=10)
        cont.pack(fill="x", pady=(0, 8))
        ttk.Label(cont, text=titulo.upper(), style="Titulo.TLabel").pack(anchor="w")
        return cont

    def _construir(self) -> None:
        raiz = ttk.Frame(self.root, padding=10)
        raiz.pack(fill="both", expand=True)

        # ---- columna izquierda: video + controles ----
        izq = ttk.Frame(raiz)
        izq.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.lienzo = tk.Canvas(izq, width=self.args.ancho, height=self.args.alto,
                                bg="#000000", highlightthickness=0)
        self.lienzo.pack(fill="both", expand=True)
        self.txt_lienzo = self.lienzo.create_text(
            self.args.ancho // 2, self.args.alto // 2,
            text="Cámara detenida", fill=MUTED, font=("Helvetica", 13))

        barra = ttk.Frame(izq, padding=(0, 10, 0, 0))
        barra.pack(fill="x")
        self.btn_iniciar = ttk.Button(barra, text="▶  Iniciar", command=self.iniciar)
        self.btn_detener = ttk.Button(barra, text="■  Detener", command=self.detener)
        self.btn_reanudar = ttk.Button(barra, text="⟳  Reanudar", command=self.reanudar)
        for b in (self.btn_iniciar, self.btn_detener, self.btn_reanudar):
            b.pack(side="left", padx=(0, 8))

        self.lbl_estado = ttk.Label(barra, text="DETENIDO", foreground=MUTED,
                                    font=("Helvetica", 11, "bold"))
        self.lbl_estado.pack(side="left", padx=12)

        self.var_mapeo = tk.StringVar(value="espejo")
        ttk.Label(barra, text="Mapeo:").pack(side="right", padx=(0, 6))
        combo = ttk.Combobox(barra, textvariable=self.var_mapeo, width=8,
                             state="readonly", values=("espejo", "dedos"))
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>",
                   lambda e: setattr(self.proc, "estrategia", self.var_mapeo.get()))

        # ---- columna derecha: paneles ----
        der = ttk.Frame(raiz, width=340)
        der.pack(side="right", fill="y")
        der.pack_propagate(False)

        # Estado del sistema
        c = self._tarjeta(der, "Estado del sistema")
        self.lbl_fps = ttk.Label(c, text="FPS —   ·   p5 —", style="Card.TLabel")
        self.lbl_fps.pack(anchor="w", pady=(4, 0))
        self.lbl_manos = ttk.Label(c, text="Manos detectadas: 0", style="Card.TLabel")
        self.lbl_manos.pack(anchor="w")
        self.lbl_orient = ttk.Label(c, text="Orientación: —", style="Muted.TLabel")
        self.lbl_orient.pack(anchor="w")

        # Postura detectada
        c = self._tarjeta(der, "Postura detectada")
        self.lbl_gesto = ttk.Label(c, text="—", style="Gesto.TLabel")
        self.lbl_gesto.pack(anchor="w", pady=(4, 2))
        self.barra_conf = ttk.Progressbar(c, maximum=100, length=300)
        self.barra_conf.pack(fill="x")
        self.lbl_detalle = ttk.Label(c, text="Sin eventos todavía",
                                     style="Muted.TLabel", wraplength=300)
        self.lbl_detalle.pack(anchor="w", pady=(4, 0))

        # Extension por dedo
        c = self._tarjeta(der, "Extensión por dedo")
        self.barras_dedo: dict[str, ttk.Progressbar] = {}
        for dedo in ("pulgar", "indice", "medio", "anular", "menique"):
            fila = ttk.Frame(c, style="Card.TFrame")
            fila.pack(fill="x", pady=1)
            ttk.Label(fila, text=dedo[:4].upper(), width=6,
                      style="Muted.TLabel").pack(side="left")
            pb = ttk.Progressbar(fila, maximum=100, length=230)
            pb.pack(side="left", fill="x", expand=True)
            self.barras_dedo[dedo] = pb

        # Matriz de sOFIa
        c = self._tarjeta(der, "sOFIa · 16 módulos")
        self.lienzo_matriz = tk.Canvas(c, width=300, height=120, bg=CARD,
                                       highlightthickness=0)
        self.lienzo_matriz.pack(pady=(4, 0))
        self.celdas = []
        cw, ch, gap = 28, 26, 4
        x0 = (300 - (GRID * cw + (GRID - 1) * gap)) // 2
        for i in range(16):
            r, col = divmod(i, GRID)
            x, y = x0 + col * (cw + gap), 6 + r * (ch + gap)
            self.celdas.append(self.lienzo_matriz.create_rectangle(
                x, y, x + cw, y + ch, fill="#2e2e38", outline="#3c3c48"))

        # Enlace con la ESP32
        c = self._tarjeta(der, "ESP32 Maestra")
        fila0 = ttk.Frame(c, style="Card.TFrame")
        fila0.pack(fill="x", pady=(4, 2))
        ttk.Label(fila0, text="Transporte:", style="Muted.TLabel").pack(side="left")
        self.transporte_var = tk.StringVar(value=self.args.transporte)
        combo_tr = ttk.Combobox(fila0, textvariable=self.transporte_var, width=7,
                                state="readonly", values=("wifi", "ble"))
        combo_tr.pack(side="left", padx=6)
        combo_tr.bind("<<ComboboxSelected>>", lambda e: self._cambiar_transporte())
        self.btn_buscar = ttk.Button(fila0, text="Buscar", width=8,
                                     command=self.buscar_ble)
        self.btn_buscar.pack(side="left")

        fila = ttk.Frame(c, style="Card.TFrame")
        fila.pack(fill="x", pady=(4, 4))
        self.ip_var = tk.StringVar(value=self.args.host or "192.168.4.1")
        self.puerto_var = tk.StringVar(value=str(self.args.puerto))
        self.ent_destino = ttk.Entry(fila, textvariable=self.ip_var, width=17)
        self.ent_destino.pack(side="left")
        self.lbl_sep = ttk.Label(fila, text=":", style="Card.TLabel")
        self.lbl_sep.pack(side="left", padx=2)
        self.ent_puerto = ttk.Entry(fila, textvariable=self.puerto_var, width=6)
        self.ent_puerto.pack(side="left")

        fila2 = ttk.Frame(c, style="Card.TFrame")
        fila2.pack(fill="x")
        self.btn_conectar = ttk.Button(fila2, text="Conectar", width=11,
                                       command=self.conectar)
        self.btn_conectar.pack(side="left", padx=(0, 6))
        self.btn_ping = ttk.Button(fila2, text="Ping", width=8, command=self.ping)
        self.btn_ping.pack(side="left")
        self.btn_stop32 = ttk.Button(fila2, text="STOP", width=7,
                                     command=self.stop_esp32)
        self.btn_stop32.pack(side="left", padx=(6, 0))

        self.lbl_enlace = ttk.Label(c, text="● Desconectado", foreground=BAD,
                                    background=CARD, font=("Helvetica", 10, "bold"))
        self.lbl_enlace.pack(anchor="w", pady=(6, 0))
        self.lbl_rtt = ttk.Label(c, text="RTT: —", style="Muted.TLabel")
        self.lbl_rtt.pack(anchor="w")

        # Registro
        c = self._tarjeta(der, "Registro")
        self.txt_log = tk.Text(c, height=7, bg="#15151a", fg=FG, bd=0,
                               font=("Menlo", 9), wrap="word")
        self.txt_log.pack(fill="both", expand=True, pady=(4, 0))
        self.txt_log.tag_config("ok", foreground=OK)
        self.txt_log.tag_config("warn", foreground=WARN)
        self.txt_log.tag_config("error", foreground=BAD)
        self.txt_log.tag_config("info", foreground=MUTED)
        self.txt_log.configure(state="disabled")

        self._botones()

    # -- utilidades -------------------------------------------------------
    def _ui(self, fn, *args) -> None:
        """
        Agenda trabajo para el hilo de Tk desde cualquier otro hilo.

        No se usa root.after() desde los hilos trabajadores: Tkinter no es
        thread-safe y esa llamada revienta con "main thread is not in main
        loop" si el bucle aun no arranco o si la ventana ya se destruyo. En
        su lugar, los hilos solo depositan en una cola y el bucle de refresco,
        que si corre en el hilo principal, la vacia. Es el unico punto de
        contacto entre hilos y widgets.
        """
        self.cola_ui.put((fn, args))

    def _vaciar_cola_ui(self) -> None:
        while True:
            try:
                fn, args = self.cola_ui.get_nowait()
            except queue.Empty:
                return
            try:
                fn(*args)
            except tk.TclError:
                return                      # la ventana ya se cerro

    def log(self, mensaje: str, nivel: str = "info") -> None:
        """Seguro desde cualquier hilo."""
        self._ui(self._log_ui, mensaje, nivel)

    def _log_ui(self, mensaje: str, nivel: str) -> None:
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"{time.strftime('%H:%M:%S')}  {mensaje}\n", nivel)
        self.txt_log.see("end")
        if int(self.txt_log.index("end-1c").split(".")[0]) > 200:
            self.txt_log.delete("1.0", "50.0")
        self.txt_log.configure(state="disabled")

    def _botones(self) -> None:
        e = self.estado
        self.btn_iniciar.state(["!disabled"] if e == "DETENIDO" else ["disabled"])
        self.btn_detener.state(["!disabled"] if e == "EJECUTANDO" else ["disabled"])
        self.btn_reanudar.state(["!disabled"] if e == "PAUSADO" else ["disabled"])
        color = {"DETENIDO": MUTED, "EJECUTANDO": OK, "PAUSADO": WARN}[e]
        self.lbl_estado.configure(text=e, foreground=color)

    # -- controles de deteccion -------------------------------------------
    def iniciar(self) -> None:
        if self.proc.error:
            self.log(f"No se puede iniciar: {self.proc.error}", "error")
            return
        self.estado = "EJECUTANDO"
        self.proc.reiniciar_detectores()
        self.proc.activo.set()
        self.log("Detección iniciada", "ok")
        self._botones()

    def detener(self) -> None:
        self.estado = "PAUSADO"
        self.proc.activo.clear()        # el hilo se bloquea en wait()
        self.log("Detección detenida. El hilo queda en espera pasiva.", "warn")
        self._botones()

    def reanudar(self) -> None:
        self.estado = "EJECUTANDO"
        # Las ventanas temporales de los gestos dinamicos quedaron con datos
        # anteriores a la pausa: se limpian para no inventar un gesto.
        self.proc.reiniciar_detectores()
        self.proc.activo.set()
        self.log("Detección reanudada; detectores dinámicos reiniciados", "ok")
        self._botones()

    # -- enlace con la ESP32 ----------------------------------------------
    def _cambiar_transporte(self) -> None:
        """
        WiFi usa IP y puerto; BLE usa el nombre anunciado del dispositivo.
        El protocolo es el mismo en ambos: solo cambia por donde viajan los
        bytes, asi que el resto de la aplicacion no se entera.
        """
        if self.enlace is not None:
            self.desconectar()
        if self.transporte_var.get() == "ble":
            self.ip_var.set("sOFIa-Maestra")
            self.lbl_sep.pack_forget()
            self.ent_puerto.pack_forget()
            self.btn_buscar.state(["!disabled"])
            self.log("Transporte BLE: el campo es el nombre del dispositivo")
        else:
            self.ip_var.set(self.args.host or "192.168.4.1")
            self.lbl_sep.pack(side="left", padx=2)
            self.ent_puerto.pack(side="left")
            self.btn_buscar.state(["disabled"])
            self.log("Transporte WiFi/UDP: el campo es la IP")

    def buscar_ble(self) -> None:
        """Escanea y rellena el campo con la primera Maestra compatible."""
        self.btn_buscar.state(["disabled"])
        self.log("Buscando dispositivos BLE (8 s)...")
        threading.Thread(target=self._buscar_worker, daemon=True).start()

    def _buscar_worker(self) -> None:
        try:
            from core.ble_link import escanear
            hallados = escanear(timeout=8.0, solo_sofia=False)
        except Exception as exc:
            self._ui(self._buscar_listo, [], str(exc))
            return
        self._ui(self._buscar_listo, hallados, None)

    def _buscar_listo(self, hallados, error) -> None:
        self.btn_buscar.state(["!disabled"])
        if error:
            self.log(f"Error al escanear BLE: {error}", "error")
            return
        compatibles = [d for d in hallados if d["compatible"]]
        self.log(f"BLE: {len(hallados)} dispositivos, "
                 f"{len(compatibles)} compatibles")
        if compatibles:
            d = compatibles[0]
            self.ip_var.set(d["nombre"])
            self.log(f"Encontrada: {d['nombre']} ({d['rssi']} dBm)", "ok")
        else:
            self.log("Ninguno anuncia el servicio del proyecto. "
                     "¿Está cargado el firmware BLE?", "warn")

    def conectar(self) -> None:
        if self.enlace is not None:
            self.desconectar()
            return
        try:
            puerto = int(self.puerto_var.get())
        except ValueError:
            self.log("Puerto inválido", "error")
            return
        destino = self.ip_var.get().strip()
        modo = self.transporte_var.get()
        self.btn_conectar.state(["disabled"])
        self.log(f"Conectando por {modo.upper()} con {destino}"
                 f"{'' if modo == 'ble' else ':' + str(puerto)} ...")
        threading.Thread(target=self._conectar_worker,
                         args=(modo, destino, puerto), daemon=True).start()

    def _conectar_worker(self, modo: str, destino: str, puerto: int) -> None:
        try:
            if modo == "ble":
                from core.ble_link import BleLink
                enlace = BleLink(nombre=destino)
            else:
                enlace = protocol.SofiaLink(destino, puerto)
            ok = enlace.hello()
        except Exception as exc:
            self._ui(self.log, f"No se pudo abrir el enlace: {exc}", "error")
            self._ui(self.btn_conectar.state, ["!disabled"])
            return
        self._ui(self._conectar_listo, enlace, ok)

    def _conectar_listo(self, enlace, ok: bool) -> None:
        self.btn_conectar.state(["!disabled"])
        if ok:
            self.enlace = enlace
            self.proc.enlace = enlace
            self.btn_conectar.configure(text="Desconectar")
            self.lbl_enlace.configure(text="● Conectado", foreground=OK)
            self.log("Handshake correcto: la ESP32 respondió ACK", "ok")
            self.ping()
        else:
            enlace.close()
            self.lbl_enlace.configure(text="● Sin respuesta", foreground=BAD)
            if self.transporte_var.get() == "ble":
                self.log("La ESP32 no respondió. Revisa el nombre, que tenga "
                         "el firmware BLE y que no esté emparejada con otro "
                         "equipo.", "error")
            else:
                self.log("La ESP32 no respondió al HELLO. Revisa IP, puerto "
                         "y red.", "error")

    def desconectar(self) -> None:
        if self.enlace is None:
            return
        try:
            self.enlace.stop()          # paro seguro antes de soltar
            self.enlace.close()
        except Exception:
            pass
        self.enlace = None
        self.proc.enlace = None
        self.btn_conectar.configure(text="Conectar")
        self.lbl_enlace.configure(text="● Desconectado", foreground=BAD)
        self.lbl_rtt.configure(text="RTT: —")
        self.log("Enlace cerrado; se envió STOP antes de soltar", "warn")

    def ping(self) -> None:
        if self.enlace is None:
            self.log("No hay enlace: pulsa Conectar primero", "warn")
            return
        self.btn_ping.state(["disabled"])
        threading.Thread(target=self._ping_worker, daemon=True).start()

    def _ping_worker(self) -> None:
        muestras = [self.enlace.ping() for _ in range(5)]
        self._ui(self._ping_listo, muestras)

    def _ping_listo(self, muestras) -> None:
        self.btn_ping.state(["!disabled"])
        buenas = [m for m in muestras if m is not None]
        perdidas = len(muestras) - len(buenas)
        if not buenas:
            self.lbl_rtt.configure(text="RTT: sin respuesta")
            self.lbl_enlace.configure(text="● Sin respuesta", foreground=BAD)
            self.log("Ping: 5 de 5 perdidos", "error")
            return
        media = sum(buenas) / len(buenas)
        self.lbl_rtt.configure(
            text=f"RTT: {media:.1f} ms  (mín {min(buenas):.1f} / "
                 f"máx {max(buenas):.1f})")
        self.lbl_enlace.configure(text="● Conectado", foreground=OK)
        nivel = "ok" if perdidas == 0 else "warn"
        self.log(f"Ping: {len(buenas)}/5 respondidos, media {media:.1f} ms", nivel)

    def stop_esp32(self) -> None:
        if self.enlace is None:
            self.log("No hay enlace", "warn")
            return
        threading.Thread(
            target=lambda: self.log(
                "STOP confirmado por la ESP32" if self.enlace.stop()
                else "STOP sin ACK", "ok" if self.enlace.connected else "error"),
            daemon=True).start()

    # -- refresco de la interfaz ------------------------------------------
    def refrescar(self) -> None:
        self._vaciar_cola_ui()          # resultados que dejaron los hilos
        try:
            c = self.cola.get_nowait()
            self.ultimo_cuadro = c
        except queue.Empty:
            c = None

        if c is not None:
            self._pintar_video(c.imagen)
            self.lbl_fps.configure(text=f"FPS {c.fps:5.1f}   ·   p5 {c.fps_p5:5.1f}")
            self.lbl_manos.configure(text=f"Manos detectadas: {c.manos}")
            self.lbl_orient.configure(
                text=f"Orientación: {c.orientacion}  ({c.palm_normal:+.2f})")
            for dedo, pb in self.barras_dedo.items():
                pb["value"] = c.extensiones.get(dedo, 0.0) * 100
            self._pintar_matriz(c.matriz)

        # Los eventos van por su propia cola: se vacian todos, aunque hayan
        # llegado varios entre dos refrescos.
        while True:
            try:
                ev = self.eventos.get_nowait()
            except queue.Empty:
                break
            self.lbl_gesto.configure(text=ev.name)
            self.barra_conf["value"] = ev.confidence * 100
            extra = ev.payload or {}
            self.lbl_detalle.configure(
                text=f"{ev.hand} · confianza {ev.confidence:.2f}"
                     + (f" · {extra}" if extra else ""))
            self.log(f"Postura: {ev.name} ({ev.confidence:.2f})", "ok")

        if c is None and self.estado == "DETENIDO" and not self.camara.ok:
            self.lienzo.itemconfigure(self.txt_lienzo,
                                      text="Buscando cámara...")

        self.root.after(20, self.refrescar)

    def _pintar_video(self, frame: np.ndarray) -> None:
        # PPM via PhotoImage: Tk lo entiende de forma nativa, asi evitamos
        # depender de Pillow solo para mostrar un cuadro.
        ok, buf = cv2.imencode(".ppm", frame)
        if not ok:
            return
        self._img_ref = tk.PhotoImage(data=buf.tobytes())
        self.lienzo.delete("video")
        self.lienzo.create_image(0, 0, anchor="nw", image=self._img_ref,
                                 tags="video")
        self.lienzo.tag_lower("video")
        self.lienzo.itemconfigure(self.txt_lienzo, text="")

    def _pintar_matriz(self, valores: np.ndarray) -> None:
        for i, v in enumerate(valores):
            t = float(np.clip(v / 255.0, 0, 1))
            r = int(46 + 200 * t)
            g = int(46 + 160 * t)
            b = int(56 + 40 * t)
            self.lienzo_matriz.itemconfigure(self.celdas[i],
                                             fill=f"#{r:02x}{g:02x}{b:02x}")

    # -- cierre -----------------------------------------------------------
    def cerrar(self) -> None:
        self.log("Cerrando...")
        if self.enlace is not None:
            try:
                self.enlace.stop()
                self.enlace.close()
            except Exception:
                pass
        self.proc.detener()
        self.camara.detener()
        self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description="GUI de control de sOFIa")
    ap.add_argument("--cam", default="auto")
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--manos", type=int, default=2)
    ap.add_argument("--host", default=None, help="IP de la ESP32 Maestra")
    ap.add_argument("--puerto", type=int, default=4210)
    ap.add_argument("--hz", type=float, default=25.0)
    ap.add_argument("--backend", choices=["auto", "legacy", "tasks"],
                    default="auto")
    ap.add_argument("--transporte", choices=["wifi", "ble"], default="wifi",
                    help="wifi = UDP por IP; ble = Bluetooth Low Energy")
    args = ap.parse_args()

    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
