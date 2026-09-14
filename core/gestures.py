"""
Reconocimiento de las 5 POSTURAS del eje tecnico SC.

El cronograma exige al menos 3 posturas DINAMICAS (que requieren mas de un
frame). El catalogo esta disenado como una "conversacion" visual con sOFIa:

  ESTATICAS
    PALMA_ABIERTA  mano abierta sostenida      -> sOFIa despliega el panel
    PUNO           mano cerrada sostenida      -> sOFIa se contrae
  DINAMICAS
    SALUDO         oscilacion lateral (adios)  -> sOFIa "responde" con una ola
    BARRIDO        desplazamiento lateral      -> onda viajera en la matriz
    PULSO          abrir-cerrar-abrir rapido   -> latido de toda la matriz

Arquitectura: cada postura es un objeto con estado propio que consume un
stream de HandFeatures y devuelve (activo, confianza). El coordinador aplica
prioridad (dinamica gana sobre estatica) y periodo refractario para que sOFIa
no reciba dos ordenes encimadas mientras los servos aun se mueven.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import HandFeatures
from .smoothing import Hysteresis, MajorityVote


@dataclass
class GestureEvent:
    name: str
    confidence: float
    hand: str
    timestamp_ms: int
    payload: dict | None = None


# ---------------------------------------------------------------------------
# Posturas estaticas: se exige permanencia (hold) para no disparar de paso
# ---------------------------------------------------------------------------
class HeldPosture:
    """
    Base: la condicion debe cumplirse durante `hold_ms` continuos Y la mano
    debe estar QUIETA.

    El requisito de quietud no es cosmetico, resuelve una ambiguedad real: un
    SALUDO tambien es, frame a frame, una mano abierta. Sin este filtro la
    postura estatica dispara primero (a los 350 ms) y el gesto dinamico nunca
    alcanza a completarse. Quieto = estatico, en movimiento = dinamico: asi el
    catalogo queda sin traslapes.

    El desplazamiento se mide en PALMAS, no en pixeles, para que el umbral no
    dependa de la distancia a la camara.
    """

    name = "BASE"

    def __init__(self, hold_ms: int = 350, max_motion_palms: float = 0.9) -> None:
        self.hold_ms = hold_ms
        self.max_motion_palms = max_motion_palms
        self._since: int | None = None
        self._track: deque[tuple[int, float, float]] = deque()

    def condition(self, f: HandFeatures) -> tuple[bool, float]:
        raise NotImplementedError

    def _is_still(self, f: HandFeatures) -> bool:
        s = max(f.scale, 1e-6)
        self._track.append((f.timestamp_ms, f.center[0] / s, f.center[1] / s))
        while self._track and f.timestamp_ms - self._track[0][0] > self.hold_ms:
            self._track.popleft()
        if len(self._track) < 3:
            return True
        xs = np.array([p[1] for p in self._track])
        ys = np.array([p[2] for p in self._track])
        span = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
        return span <= self.max_motion_palms

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        ok, conf = self.condition(f)
        still = self._is_still(f)
        if not ok or not still:
            self._since = None
            return False, conf
        if self._since is None:
            self._since = f.timestamp_ms
        held = f.timestamp_ms - self._since
        return held >= self.hold_ms, conf

    def reset(self) -> None:
        self._since = None
        self._track.clear()


class PalmaAbierta(HeldPosture):
    name = "PALMA_ABIERTA"

    def condition(self, f: HandFeatures) -> tuple[bool, float]:
        ok = f.aperture > 0.72 and f.n_fingers_up >= 4 and f.spread > 0.45
        return ok, float(np.clip(f.aperture, 0, 1))


class Puno(HeldPosture):
    name = "PUNO"

    def condition(self, f: HandFeatures) -> tuple[bool, float]:
        ok = f.aperture < 0.28 and f.n_fingers_up <= 1
        return ok, float(np.clip(1.0 - f.aperture, 0, 1))


# ---------------------------------------------------------------------------
# Posturas dinamicas: operan sobre una ventana temporal
# ---------------------------------------------------------------------------
class Saludo:
    """
    SALUDO (adios): la mano esta abierta y el centro oscila lateralmente.

    Deteccion: se cuentan los cruces por cero de la velocidad horizontal dentro
    de una ventana de ~1.2 s. Un saludo real produce >= 3 inversiones con una
    amplitud minima. Contar inversiones, y no solo "se movio", es lo que separa
    un saludo de un simple desplazamiento de la mano.
    """

    name = "SALUDO"

    def __init__(self, window_ms: int = 1200, min_reversals: int = 3,
                 min_amplitude_palms: float = 1.1) -> None:
        self.window_ms = window_ms
        self.min_reversals = min_reversals
        self.min_amplitude = min_amplitude_palms
        self.buf: deque[tuple[int, float]] = deque()

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        if f.aperture < 0.55:                    # solo se saluda con mano abierta
            self.buf.clear()
            return False, 0.0

        # posicion en PALMAS: el umbral vale igual a 1 m o a 4 m de la camara
        self.buf.append((f.timestamp_ms, float(f.center[0] / max(f.scale, 1e-6))))
        while self.buf and f.timestamp_ms - self.buf[0][0] > self.window_ms:
            self.buf.popleft()
        if len(self.buf) < 8:
            return False, 0.0

        xs = np.array([x for _, x in self.buf])
        amplitude = float(xs.max() - xs.min())
        if amplitude < self.min_amplitude:
            return False, 0.0

        v = np.diff(xs)
        v = v[np.abs(v) > 1e-4]                  # descartar micro-ruido
        if len(v) < 3:
            return False, 0.0
        reversals = int(np.sum(np.sign(v[:-1]) != np.sign(v[1:])))

        ok = reversals >= self.min_reversals
        conf = float(np.clip(reversals / (self.min_reversals + 2), 0, 1))
        if ok:
            self.buf.clear()                     # consumir el evento
        return ok, conf

    def reset(self) -> None:
        self.buf.clear()


class Barrido:
    """
    BARRIDO: desplazamiento lateral neto, rapido y MONOTONO (sin regresar).
    Es el complemento del saludo: misma familia de senal, criterio opuesto.
    Devuelve la direccion en el payload para que sOFIa propague la onda.
    """

    name = "BARRIDO"

    def __init__(self, window_ms: int = 600, min_travel_palms: float = 1.6) -> None:
        self.window_ms = window_ms
        self.min_travel = min_travel_palms
        self.buf: deque[tuple[int, float]] = deque()
        self.direction = 0

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        if f.aperture < 0.5:
            self.buf.clear()
            return False, 0.0

        self.buf.append((f.timestamp_ms, float(f.center[0] / max(f.scale, 1e-6))))
        while self.buf and f.timestamp_ms - self.buf[0][0] > self.window_ms:
            self.buf.popleft()
        if len(self.buf) < 6:
            return False, 0.0

        xs = np.array([x for _, x in self.buf])
        net = float(xs[-1] - xs[0])
        path = float(np.abs(np.diff(xs)).sum())
        monotonia = abs(net) / max(path, 1e-6)   # 1.0 = linea recta sin retorno

        ok = abs(net) > self.min_travel and monotonia > 0.8
        if ok:
            self.direction = 1 if net > 0 else -1
            self.buf.clear()
        return ok, float(np.clip(monotonia, 0, 1))

    def reset(self) -> None:
        self.buf.clear()


class Pulso:
    """
    PULSO: abrir -> cerrar -> abrir en menos de `max_ms`.
    Maquina de estados de 3 etapas sobre la apertura global, con histeresis en
    cada transicion. Es el gesto mas "conversacional": equivale a un guino.
    """

    name = "PULSO"

    def __init__(self, max_ms: int = 1100) -> None:
        self.max_ms = max_ms
        self.state = 0                  # 0 esperando abierta, 1 cerro, 2 reabrio
        self.t0 = 0
        self.open_h = Hysteresis(on=0.68, off=0.55, state=False)
        self.close_h = Hysteresis(on=0.72, off=0.55, state=False)  # sobre 1-apertura

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        is_open = self.open_h(f.aperture)
        is_closed = self.close_h(1.0 - f.aperture)

        if self.state > 0 and f.timestamp_ms - self.t0 > self.max_ms:
            self.state = 0               # se agoto el tiempo, reiniciar

        if self.state == 0 and is_open:
            self.state, self.t0 = 1, f.timestamp_ms
        elif self.state == 1 and is_closed:
            self.state = 2
        elif self.state == 2 and is_open:
            dur = f.timestamp_ms - self.t0
            self.state = 0
            return True, float(np.clip(1.0 - dur / self.max_ms, 0.3, 1.0))
        return False, 0.0

    def reset(self) -> None:
        self.state = 0


# ---------------------------------------------------------------------------
# Coordinador
# ---------------------------------------------------------------------------
class GestureEngine:
    """
    Mantiene una instancia de cada detector POR MANO rastreada y resuelve
    conflictos. Regla de prioridad: dinamico > estatico, y luego confianza.
    El periodo refractario evita saturar a sOFIa: los servos tardan cientos de
    milisegundos en ejecutar un patron, no tiene sentido mandar otro antes.
    """

    PRIORITY = ["PULSO", "SALUDO", "BARRIDO", "PUNO", "PALMA_ABIERTA"]

    def __init__(self, refractory_ms: int = 900) -> None:
        self.refractory_ms = refractory_ms
        self._banks: dict[str, dict] = {}
        self._last_emit: dict[str, int] = {}
        self._vote = MajorityVote(window=5, ratio=0.6)

    def _bank(self, key: str) -> dict:
        if key not in self._banks:
            self._banks[key] = {
                "PALMA_ABIERTA": PalmaAbierta(),
                "PUNO": Puno(),
                "SALUDO": Saludo(),
                "BARRIDO": Barrido(),
                "PULSO": Pulso(),
            }
        return self._banks[key]

    def update(self, key: str, f: HandFeatures) -> GestureEvent | None:
        bank = self._bank(key)
        hits: list[tuple[str, float]] = []
        for name, det in bank.items():
            ok, conf = det.update(f)
            if ok:
                hits.append((name, conf))

        if not hits:
            self._vote("NINGUNO")
            return None

        hits.sort(key=lambda h: (self.PRIORITY.index(h[0]), -h[1]))
        name, conf = hits[0]

        last = self._last_emit.get(key, -10 ** 9)
        if f.timestamp_ms - last < self.refractory_ms:
            return None
        self._last_emit[key] = f.timestamp_ms

        payload = None
        if name == "BARRIDO":
            payload = {"direccion": bank["BARRIDO"].direction}
        elif name in ("PALMA_ABIERTA", "PUNO"):
            payload = {"apertura": round(f.aperture, 3), "dedos": f.code}

        return GestureEvent(name, conf, f.handedness, f.timestamp_ms, payload)

    def drop(self, key: str) -> None:
        """Olvidar una mano que dejo de verse, para no arrastrar estado viejo."""
        self._banks.pop(key, None)
        self._last_emit.pop(key, None)
