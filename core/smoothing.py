"""
Suavizado y estabilizacion (requisito explicito de la semana 6 del cronograma).

Hay TRES fuentes de inestabilidad distintas y cada una necesita su propia
herramienta. Confundirlas es el error clasico:

  1. Jitter de landmarks (el punto vibra 2-3 px entre frames)
     -> filtro One Euro sobre coordenadas.
  2. Parpadeo de clasificacion (el gesto salta PUNO/PALMA/PUNO)
     -> voto por mayoria en ventana + histeresis de entrada/salida.
  3. Chattering mecanico (el servo tiembla por micro-cambios del comando)
     -> banda muerta + limitador de velocidad (slew rate).

El 3 es el que mas importa aqui: sOFIa es hardware. Un comando ruidoso no solo
se ve mal, desgasta servos y satura el bus.
"""

from __future__ import annotations

import math
from collections import Counter, deque

import numpy as np


class LowPass:
    """Filtro pasa-bajas exponencial de primer orden."""

    def __init__(self) -> None:
        self.y_prev: np.ndarray | None = None

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.y_prev is None:
            self.y_prev = x
        else:
            self.y_prev = alpha * x + (1.0 - alpha) * self.y_prev
        return self.y_prev

    def reset(self) -> None:
        self.y_prev = None


class OneEuroFilter:
    """
    Filtro One Euro (Casiez et al., 2012).

    Por que no un promedio movil: un promedio movil de N frames introduce
    siempre N/2 frames de retardo, lo que se siente como "la mano va adelantada
    respecto al esqueleto". One Euro adapta su corte a la velocidad: filtra
    fuerte cuando la mano esta quieta (mata el jitter) y deja pasar cuando la
    mano se mueve rapido (no introduce lag perceptible). Es el compromiso
    correcto para un sistema interactivo en tiempo real.

    Parametros:
        min_cutoff : baja -> mas suave en reposo, mas lag.
        beta       : sube -> menos lag en movimiento rapido, mas jitter.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = LowPass()
        self._dx = LowPass()
        self._t_prev: float | None = None
        self._x_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: np.ndarray, t_seconds: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if self._t_prev is None:
            self._t_prev = t_seconds
            self._x_prev = x
            self._x(x, 1.0)
            return x

        dt = max(t_seconds - self._t_prev, 1e-3)
        self._t_prev = t_seconds

        dx = (x - self._x_prev) / dt
        self._x_prev = x
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        return self._x(x, self._alpha(float(cutoff), dt))

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._t_prev = None
        self._x_prev = None


class LandmarkSmoother:
    """Aplica One Euro a los 21 landmarks (x, y, z) de una mano."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02) -> None:
        self.f = OneEuroFilter(min_cutoff, beta)

    def __call__(self, pts: np.ndarray, t_seconds: float) -> np.ndarray:
        flat = self.f(np.asarray(pts, dtype=float).ravel(), t_seconds)
        return flat.reshape(-1, 3)

    def reset(self) -> None:
        self.f.reset()


class MajorityVote:
    """
    Voto por mayoria sobre una ventana deslizante de etiquetas.
    Una etiqueta solo se acepta si domina al menos `ratio` de la ventana.
    Elimina el parpadeo de un frame sin retrasar mas que `window` frames.
    """

    def __init__(self, window: int = 7, ratio: float = 0.6) -> None:
        self.buf: deque[str] = deque(maxlen=window)
        self.ratio = ratio
        self.stable = "NINGUNO"

    def __call__(self, label: str) -> str:
        self.buf.append(label)
        if len(self.buf) < self.buf.maxlen:
            return self.stable
        top, n = Counter(self.buf).most_common(1)[0]
        if n / len(self.buf) >= self.ratio:
            self.stable = top
        return self.stable

    def reset(self) -> None:
        self.buf.clear()
        self.stable = "NINGUNO"


class Hysteresis:
    """
    Umbral con histeresis (disparador de Schmitt).
    Entrar cuesta mas que salir: evita el rebote cuando la senal se queda
    justo sobre el umbral. `on` debe ser mayor que `off`.
    """

    def __init__(self, on: float = 0.6, off: float = 0.4, state: bool = False):
        self.on, self.off, self.state = on, off, state

    def __call__(self, value: float) -> bool:
        if self.state:
            if value < self.off:
                self.state = False
        elif value > self.on:
            self.state = True
        return self.state


class SlewLimiter:
    """
    Limita cuanto puede cambiar un comando por segundo y aplica banda muerta.

    Protege el hardware: sin esto, un landmark ruidoso se traduce en el servo
    persiguiendo microcambios a 30 Hz (ruido audible, calentamiento, desgaste)
    y en trafico innecesario hacia la ESP32.

        max_rate : unidades (0-255) por segundo
        deadband : cambio minimo para que valga la pena mover
    """

    def __init__(self, size: int = 16, max_rate: float = 420.0,
                 deadband: float = 4.0) -> None:
        self.value = np.zeros(size, dtype=float)
        self.max_rate = max_rate
        self.deadband = deadband
        self._t_prev: float | None = None

    def __call__(self, target: np.ndarray, t_seconds: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if self._t_prev is None:
            self._t_prev = t_seconds
            self.value = target.copy()
            return self.value

        dt = max(t_seconds - self._t_prev, 1e-3)
        self._t_prev = t_seconds

        delta = target - self.value
        delta[np.abs(delta) < self.deadband] = 0.0          # banda muerta
        max_step = self.max_rate * dt
        delta = np.clip(delta, -max_step, max_step)          # limite de rapidez
        self.value = self.value + delta
        return self.value
