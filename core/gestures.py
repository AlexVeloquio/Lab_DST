"""
Reconocimiento de las 5 POSTURAS del eje tecnico SC. Version final.

CATALOGO
--------
  DINAMICAS (3, el minimo que exige el cronograma)
    SALUDO   oscilacion lateral de la mano abierta   -> sOFIa devuelve una ola
    PULSO    abrir - cerrar - abrir rapido           -> latido de toda la matriz
    GIRO     voltear de palma a dorso o al reves     -> la matriz se espeja

  ESTATICAS (2)
    PUNO     mano cerrada, quieta                    -> sOFIa se contrae
    PAZ      indice y medio arriba, quieta           -> patron en V

PRINCIPIO DE DISENO: CADA POSTURA EN SU PROPIO EJE DE SEÑAL
------------------------------------------------------------
Dos posturas solo pueden confundirse si compiten por la misma medida. El
catalogo esta armado para que cada una viva en un eje distinto:

    SALUDO  ->  desplazamiento lateral del centro
    PULSO   ->  apertura global variando en el tiempo
    GIRO    ->  normal de la palma cruzando de signo
    PUNO    ->  apertura global baja, sostenida
    PAZ     ->  PATRON de dedos (cuales, no cuantos), sostenido

`PAZ` es la pieza que cierra el diseño: usa el patron de dedos, un eje que
ninguna otra postura toca. Su apertura ronda 0.43, por debajo del umbral de
0.55 que exige SALUDO, asi que es imposible que una se lea como la otra.

QUE SE QUITO Y POR QUE
----------------------
  PALMA_ABIERTA  competia directamente con SALUDO: una mano saludando ES una
                 mano abierta frame a frame, y una mano abierta quieta caia en
                 el detector de saludo cuando el jitter de landmarks alcanzaba
                 para simular oscilaciones. Se sustituyo por PAZ, que no
                 comparte eje con nada.
  BARRIDO        usaba exactamente la misma señal que SALUDO (desplazamiento
                 lateral) con el criterio opuesto. Mantener las dos obligaba a
                 umbrales tan finos que cualquier movimiento intermedio caia en
                 tierra de nadie.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import HandFeatures
from .smoothing import Hysteresis


@dataclass
class GestureEvent:
    name: str
    confidence: float
    hand: str
    timestamp_ms: int
    payload: dict | None = None


# ---------------------------------------------------------------------------
# Posturas estaticas
# ---------------------------------------------------------------------------
class HeldPosture:
    """
    Base: la condicion debe cumplirse durante `hold_ms` continuos Y la mano
    debe estar QUIETA.

    "Quieta" son TRES cosas, no una: sin desplazamiento en x, sin
    desplazamiento en y y SIN GIRAR. Cada una se agrego para resolver una
    colision concreta con una postura dinamica, porque frame a frame una mano
    que saluda o que voltea cumple igual de bien la condicion estatica. Quieto
    = estatico, en movimiento = dinamico: asi el catalogo no se traslapa.

    El desplazamiento se mide en PALMAS, no en pixeles, para que el umbral no
    dependa de la distancia a la camara.
    """

    name = "BASE"

    def __init__(self, hold_ms: int = 400, max_motion_palms: float = 0.8,
                 max_giro: float = 0.5) -> None:
        self.hold_ms = hold_ms
        self.max_motion_palms = max_motion_palms
        self.max_giro = max_giro
        self._since: int | None = None
        self._track: deque[tuple[int, float, float, float]] = deque()

    def condition(self, f: HandFeatures) -> tuple[bool, float]:
        raise NotImplementedError

    def _is_still(self, f: HandFeatures) -> bool:
        s = max(f.scale, 1e-6)
        self._track.append((f.timestamp_ms, f.center[0] / s, f.center[1] / s,
                            f.palm_normal))
        while self._track and f.timestamp_ms - self._track[0][0] > self.hold_ms:
            self._track.popleft()
        if len(self._track) < 3:
            return True

        xs = np.array([p[1] for p in self._track])
        ys = np.array([p[2] for p in self._track])
        vs = np.array([p[3] for p in self._track])

        desplazo = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
        giro = float(vs.max() - vs.min())
        return desplazo <= self.max_motion_palms and giro <= self.max_giro

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        ok, conf = self.condition(f)
        if not ok or not self._is_still(f):
            self._since = None
            return False, conf
        if self._since is None:
            self._since = f.timestamp_ms
        return (f.timestamp_ms - self._since) >= self.hold_ms, conf

    def reset(self) -> None:
        self._since = None
        self._track.clear()


class Puno(HeldPosture):
    """
    PUNO: mano cerrada por completo.

    Se exige `n_fingers_up == 0`, no "uno o menos". La diferencia importa: con
    el criterio flojo, señalar con el indice tambien contaba como puño, y eso
    cierra la puerta a cualquier postura futura basada en un solo dedo.
    """

    name = "PUNO"

    def condition(self, f: HandFeatures) -> tuple[bool, float]:
        ok = f.aperture < 0.25 and f.n_fingers_up == 0
        return ok, float(np.clip(1.0 - f.aperture * 2.0, 0.0, 1.0))


class Paz(HeldPosture):
    """
    PAZ: indice y medio extendidos, anular y meñique recogidos.

    El pulgar se deja libre a proposito: mucha gente hace la V con el pulgar
    pegado y mucha con el pulgar afuera, y exigirlo en una posicion concreta
    solo produciria rechazos que el usuario no entiende.

    Se pide explicitamente que anular y meñique esten BAJO 0.4 de extension, no
    solo "no arriba". El umbral de `fingers_up` es 0.55, asi que un dedo a
    medio camino (0.5) contaria como abajo sin estarlo; esa banda intermedia es
    justo donde vive una mano que esta abriendo o cerrando, y sin el segundo
    umbral la V se dispararia a media transicion de un PULSO.
    """

    name = "PAZ"

    def condition(self, f: HandFeatures) -> tuple[bool, float]:
        e = f.extensions
        arriba = e["indice"] > 0.6 and e["medio"] > 0.6
        abajo = e["anular"] < 0.4 and e["menique"] < 0.4
        ok = arriba and abajo

        margen = min(e["indice"], e["medio"]) - max(e["anular"], e["menique"])
        return ok, float(np.clip(margen, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Posturas dinamicas
# ---------------------------------------------------------------------------
class Saludo:
    """
    SALUDO (adios): la mano abierta oscila lateralmente.

    ALGORITMO: zigzag con amplitud minima por oscilacion.

    La version ingenua (contar cambios de signo de la velocidad) NO sirve, y
    fue la causa del bug que hacia que una palma quieta se leyera como saludo:
    el jitter de landmarks produce decenas de cambios de signo por segundo, y
    aunque sean de 0.02 palmas, cuentan igual que un saludo real.

    Aqui una oscilacion solo cuenta si el recorrido desde el ultimo extremo
    supera `min_swing` palmas. Una mano quieta produce CERO oscilaciones por
    mucho que tiemble, porque nunca se aleja tanto de su extremo. Ademas la
    serie se suaviza con una media movil corta antes de analizarla.

    Todo se mide en PALMAS, asi que el umbral vale igual a 1 m que a 4 m.
    """

    name = "SALUDO"

    def __init__(self, window_ms: int = 1400, min_swings: int = 3,
                 min_amplitude_palms: float = 1.2,
                 min_swing_palms: float = 0.45) -> None:
        self.window_ms = window_ms
        self.min_swings = min_swings
        self.min_amplitude = min_amplitude_palms
        self.min_swing = min_swing_palms
        self.buf: deque[tuple[int, float]] = deque()

    @staticmethod
    def _count_swings(xs: np.ndarray, min_swing: float) -> int:
        """
        Cuenta inversiones de direccion que superen `min_swing`.
        Es el mismo principio de un indicador zigzag: los tramos pequeños se
        ignoran por completo en lugar de suavizarse.
        """
        swings = 0
        ext = float(xs[0])
        direccion = 0                       # 0 sin definir, +1 sube, -1 baja

        for x in xs[1:]:
            x = float(x)
            if direccion == 0:
                if abs(x - ext) >= min_swing:
                    direccion = 1 if x > ext else -1
                    ext = x
                continue

            if (x - ext) * direccion > 0:
                ext = x                     # el movimiento actual continua
            elif abs(x - ext) >= min_swing:
                swings += 1                 # inversion significativa
                direccion = -direccion
                ext = x
        return swings

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        if f.aperture < 0.55:               # solo se saluda con la mano abierta
            self.buf.clear()
            return False, 0.0

        self.buf.append((f.timestamp_ms, float(f.center[0] / max(f.scale, 1e-6))))
        while self.buf and f.timestamp_ms - self.buf[0][0] > self.window_ms:
            self.buf.popleft()
        if len(self.buf) < 10:
            return False, 0.0

        xs = np.array([x for _, x in self.buf])
        if len(xs) >= 5:                    # media movil corta contra el jitter
            xs = np.convolve(xs, np.ones(3) / 3.0, mode="valid")

        if float(xs.max() - xs.min()) < self.min_amplitude:
            return False, 0.0

        swings = self._count_swings(xs, self.min_swing)
        if swings < self.min_swings:
            return False, 0.0

        self.buf.clear()                    # consumir el evento
        return True, float(np.clip(swings / (self.min_swings + 2), 0.4, 1.0))

    def reset(self) -> None:
        self.buf.clear()


class Pulso:
    """
    PULSO: abrir -> cerrar -> abrir en menos de `max_ms`.

    Maquina de 3 estados sobre la apertura global, con histeresis en cada
    transicion para que la señal no rebote en los umbrales. Es el gesto mas
    conversacional del catalogo: equivale a un guiño.
    """

    name = "PULSO"

    def __init__(self, max_ms: int = 1400) -> None:
        self.max_ms = max_ms
        self.state = 0                  # 0 espera abierta, 1 cerro, 2 reabrio
        self.t0 = 0
        self.open_h = Hysteresis(on=0.68, off=0.55, state=False)
        self.close_h = Hysteresis(on=0.72, off=0.55, state=False)   # sobre 1-ap

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        is_open = self.open_h(f.aperture)
        is_closed = self.close_h(1.0 - f.aperture)

        if self.state > 0 and f.timestamp_ms - self.t0 > self.max_ms:
            self.state = 0                  # se agoto el tiempo, reiniciar

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


class Giro:
    """
    GIRO: la mano voltea de PALMA a DORSO o al reves.

    Señal: `palm_normal`, la componente z de la normal de la palma, continua en
    [-1, +1]. Vale +1 con la palma de frente, -1 con el dorso de frente y ~0
    con la mano de canto.

    Tres decisiones que lo hacen fiable:

    1. ZONA MUERTA AMPLIA (banda de canto). Solo se "compromete" un estado
       cuando |v| supera `commit`. Entre -commit y +commit la mano esta de
       canto: ahi la normal es casi perpendicular al eje optico y su SIGNO se
       vuelve ruido, brincando entre +1 y -1 de un frame a otro. Creerle al
       signo en esa zona genera giros fantasma sin parar.

    2. HAY QUE PASAR POR EL CANTO. Un volteo fisico obliga a cruzar la banda
       muerta. Un salto directo de +1 a -1 sin pasar por ella no es un giro: es
       un parpadeo del clasificador de mano (si MediaPipe confunde izquierda
       con derecha en un frame, la normal se invierte de golpe porque el
       producto cruz cambia de signo). Exigir el paso por el canto elimina ese
       falso positivo de raiz.

    3. EL RELOJ MIDE LA TRANSICION, NO LA ESPERA. `max_ms` se cuenta desde que
       la mano ABANDONA un estado comprometido, no desde que lo alcanzo. Asi
       puedes tener la palma quieta diez segundos y luego voltear: lo que debe
       ser rapido es el volteo, no la pausa previa.

    Se exige la mano razonablemente abierta: con el puño cerrado el arco de
    nudillos se comprime, la base canonica se vuelve inestable y la normal deja
    de significar algo.

    El detector busca un CAMBIO de signo, no una orientacion absoluta. Eso lo
    vuelve inmune a que la convencion de signo este invertida por el cv2.flip()
    del espejo o por el etiquetado de handedness: si algun dia la direccion sale
    al reves, se invierte `direction` sin tocar la deteccion.
    """

    name = "GIRO"

    def __init__(self, commit: float = 0.50, max_ms: int = 1500,
                 settle_ms: int = 180, min_aperture: float = 0.40) -> None:
        self.commit = commit
        self.max_ms = max_ms
        self.settle_ms = settle_ms
        self.min_aperture = min_aperture
        self.estado: int | None = None      # +1 palma, -1 dorso, None sin definir
        self.t_commit = 0
        self.t_salida: int | None = None    # cuando entro a la banda de canto
        self.direction = 0                  # +1 quedo en palma, -1 quedo en dorso

    def update(self, f: HandFeatures) -> tuple[bool, float]:
        if f.aperture < self.min_aperture:
            self.reset()
            return False, 0.0

        v = f.palm_normal
        ts = f.timestamp_ms

        if v > self.commit:
            nuevo = 1
        elif v < -self.commit:
            nuevo = -1
        else:
            nuevo = None                     # mano de canto: en transicion

        if nuevo is None:
            if self.estado is not None and self.t_salida is None:
                if ts - self.t_commit >= self.settle_ms:
                    self.t_salida = ts       # arranca el cronometro del volteo
                else:
                    self.estado = None       # no duro lo suficiente, se descarta
            return False, 0.0

        if self.estado is None:
            self.estado, self.t_commit, self.t_salida = nuevo, ts, None
            return False, 0.0

        if nuevo == self.estado:
            self.t_salida = None             # volvio al mismo lado, no fue giro
            return False, 0.0

        paso_por_canto = self.t_salida is not None
        dur = ts - self.t_salida if paso_por_canto else self.max_ms + 1
        ok = paso_por_canto and dur <= self.max_ms

        self.estado, self.t_commit, self.t_salida = nuevo, ts, None
        if not ok:
            return False, 0.0

        self.direction = nuevo
        return True, float(np.clip(1.0 - dur / self.max_ms, 0.35, 1.0))

    def reset(self) -> None:
        self.estado = None
        self.t_salida = None


# ---------------------------------------------------------------------------
# Coordinador
# ---------------------------------------------------------------------------
class GestureEngine:
    """
    Mantiene una instancia de cada detector POR MANO rastreada y resuelve
    conflictos. Regla de prioridad: dinamico > estatico, y luego confianza.

    El periodo refractario evita saturar a sOFIa: los servos tardan cientos de
    milisegundos en ejecutar un patron, no tiene sentido mandar otro antes.

    EXCEPCION POR PRIORIDAD: un gesto de MAYOR prioridad si puede interrumpir
    el refractario de uno menor. Una estatica nunca interrumpe a una dinamica,
    y dos gestos del mismo nivel respetan el refractario completo.
    """

    PRIORITY = ["PULSO", "GIRO", "SALUDO", "PUNO", "PAZ"]

    def __init__(self, refractory_ms: int = 900) -> None:
        self.refractory_ms = refractory_ms
        self._banks: dict[str, dict] = {}
        self._last_emit: dict[str, tuple[int, str]] = {}

    def _bank(self, key: str) -> dict:
        if key not in self._banks:
            self._banks[key] = {
                "PUNO": Puno(),
                "PAZ": Paz(),
                "SALUDO": Saludo(),
                "PULSO": Pulso(),
                "GIRO": Giro(),
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
            return None

        hits.sort(key=lambda h: (self.PRIORITY.index(h[0]), -h[1]))
        name, conf = hits[0]

        last = self._last_emit.get(key)
        if last is not None:
            ts_last, name_last = last
            dentro = f.timestamp_ms - ts_last < self.refractory_ms
            escala = self.PRIORITY.index(name) < self.PRIORITY.index(name_last)
            if dentro and not escala:
                return None
        self._last_emit[key] = (f.timestamp_ms, name)

        payload = None
        if name == "GIRO":
            d = bank["GIRO"].direction
            payload = {"direccion": d, "hacia": "PALMA" if d > 0 else "DORSO"}
        elif name in ("PUNO", "PAZ"):
            payload = {"apertura": round(f.aperture, 3), "dedos": f.code}

        return GestureEvent(name, conf, f.handedness, f.timestamp_ms, payload)

    def drop(self, key: str) -> None:
        """Olvidar una mano que dejo de verse, para no arrastrar estado viejo."""
        self._banks.pop(key, None)
        self._last_emit.pop(key, None)
