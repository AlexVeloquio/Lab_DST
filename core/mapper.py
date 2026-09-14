"""
Mapeo mano -> 16 modulos de sOFIa.

Aqui esta el corazon del proyecto: MediaPipe da 21 puntos y sOFIa tiene 16
modulos. No hay correspondencia natural, hay que DISENARLA. Se implementan dos
estrategias porque responden a intenciones distintas, y conviene defenderlas
asi en el reporte:

  ESPEJO (4x4 espacial)
      La matriz se comporta como una sombra de la mano. Se rasteriza la nube
      densa de puntos sobre una retícula de 4x4 y cada modulo recibe la
      densidad de su celda. Legible para el publico: "la pared imita mi mano".

  DEDOS (5x3 + 1 semantico)
      15 modulos = 5 dedos x 3 niveles de flexion (barra de progreso por dedo)
      + 1 modulo de apertura global. No imita la forma, comunica el ESTADO de
      cada dedo. Es el mapeo correcto cuando lo que importa es que se lea que
      dedo se movio, no donde estaba la mano.

Ambas devuelven un vector de 16 enteros 0-255 listo para la trama MATRIX16.

Convencion de indices (fila mayor, vista desde el espectador):

        col0  col1  col2  col3
  fila0   0     1     2     3
  fila1   4     5     6     7
  fila2   8     9    10    11
  fila3  12    13    14    15
"""

from __future__ import annotations

import numpy as np

from . import geometry as G
from . import landmarks as LM
from .features import HandFeatures

GRID = 4
N_MODULES = GRID * GRID


# ---------------------------------------------------------------------------
# Estrategia ESPEJO
# ---------------------------------------------------------------------------
def espejo(
    f: HandFeatures,
    sigma: float = 0.38,
    per_bone: int = 8,
    mirror: bool = True,
    canonical: bool = False,
) -> np.ndarray:
    """
    Rasteriza la mano sobre la retícula 4x4.

    sigma      ancho del nucleo gaussiano en unidades de celda. Chico = dedos
               nitidos pero matriz casi apagada; grande = mancha difusa donde
               todos los gestos se ven iguales. 0.38 es el valor donde se
               mantiene el contraste entre mano abierta, puno y senalar con
               una retícula tan gruesa como 4x4.
    mirror     True para que sOFIa actue como espejo (la mano derecha del
               usuario ilumina el lado derecho visto por el usuario).
    canonical  True elimina la rotacion de la mano: el gesto se ve igual aunque
               la persona incline la muneca. False conserva la orientacion real,
               que es lo que hace legible el efecto de espejo. Por defecto False.
    """
    pts = f.points.astype(float)

    if canonical:
        pts = G.to_canonical(pts)[:, :2]
        pts = pts * 0.5 + 0.5                      # llevar a [0,1] aprox
    else:
        pts = pts[:, :2]

    dense = G.densify(
        np.column_stack([pts, np.zeros(len(pts))]), per_bone=per_bone
    )[:, :2]

    # Escalar por el TAMANO DE PALMA, no por la caja delimitadora de la mano.
    #
    # Esta decision importa: si se normaliza por la caja, el puno se estira
    # hasta llenar la retícula igual que la mano abierta y ambos gestos se ven
    # identicos en la pared. Usando la palma como unidad, la mano abierta ocupa
    # los 4x4 y el puno se concentra al centro, que es la lectura que uno
    # espera de una sombra.
    palm = float(np.linalg.norm(pts[LM.MIDDLE_MCP] - pts[LM.WRIST]))
    palm = max(palm, 1e-6)
    center = dense.mean(axis=0)                    # centro visual de la mano
    norm = (dense - center) / (palm * 4.0) + 0.5   # -> aprox [0,1]
    norm = np.clip(norm, -0.25, 1.25)

    if mirror:
        norm[:, 0] = 1.0 - norm[:, 0]

    # Centros de celda
    idx = (np.arange(GRID) + 0.5) / GRID
    cx, cy = np.meshgrid(idx, idx)
    centers = np.column_stack([cx.ravel(), cy.ravel()])   # (16,2) fila mayor

    # Acumulacion gaussiana: cada punto reparte energia a las celdas cercanas
    d2 = ((norm[:, None, :] - centers[None, :, :]) * GRID) ** 2
    w = np.exp(-d2.sum(axis=2) / (2.0 * sigma ** 2))
    acc = w.sum(axis=0)

    # COBERTURA, no densidad.
    #
    # Si se usara densidad cruda, el puno (muchos puntos apretados en pocas
    # celdas) iluminaria MAS que la mano abierta, que es justo lo contrario de
    # lo que la gente espera ver. La saturacion exponencial convierte la medida
    # en "esta celda esta ocupada por la mano": abrir la mano ocupa mas celdas
    # y por lo tanto enciende mas modulos.
    #
    # Normalizar por acc.max() tampoco sirve: obligaria a que SIEMPRE haya un
    # modulo al 100% y el brillo absoluto dejaria de significar algo.
    k = max(per_bone * 1.2, 0.5)
    return (1.0 - np.exp(-acc / k)) * 255.0


# ---------------------------------------------------------------------------
# Estrategia DEDOS
# ---------------------------------------------------------------------------
def dedos(f: HandFeatures) -> np.ndarray:
    """
    5 columnas (un dedo cada una) x 3 filas (niveles de flexion) + 1 global.

    Cada dedo se dibuja como una barra que crece de abajo hacia arriba conforme
    se extiende. El nivel l se enciende proporcionalmente segun la extension
    rebase el tercio correspondiente, lo que da una transicion continua en vez
    de un encendido binario: los servos se mueven suave.

    Layout (5 dedos x 3 niveles = 15, indices 0..14) + modulo 15 = apertura.
    """
    out = np.zeros(N_MODULES, dtype=float)

    for c, finger in enumerate(LM.FINGER_NAMES):
        ext = f.extensions[finger]
        for level in range(3):
            lo = level / 3.0
            val = float(np.clip((ext - lo) * 3.0, 0.0, 1.0))
            # fila 0 = punta del dedo (nivel 2), fila 2 = base (nivel 0)
            row = 2 - level
            out[row * 5 + c] = val * 255.0

    out[15] = float(np.clip(f.aperture, 0.0, 1.0)) * 255.0
    return out


# ---------------------------------------------------------------------------
# Fachada
# ---------------------------------------------------------------------------
class ModuleMapper:
    """
    Selector de estrategia + fusion de varias manos.

    Con dos manos en escena se toma el MAXIMO por modulo, no el promedio: el
    promedio apagaria la mano mas expresiva y el resultado se ve muerto.
    """

    STRATEGIES = ("espejo", "dedos")

    def __init__(self, strategy: str = "espejo", **kwargs) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"estrategia desconocida: {strategy}")
        self.strategy = strategy
        self.kwargs = kwargs

    def __call__(self, hands: list[HandFeatures]) -> np.ndarray:
        if not hands:
            return np.zeros(N_MODULES, dtype=float)

        fn = espejo if self.strategy == "espejo" else dedos
        mats = [fn(h, **self.kwargs) if fn is espejo else fn(h) for h in hands]
        return np.max(np.stack(mats), axis=0)


def to_bytes(values: np.ndarray) -> bytes:
    """Cuantiza a 16 bytes 0-255 para la trama MATRIX16."""
    v = np.clip(np.asarray(values, dtype=float), 0, 255).astype(np.uint8)
    if v.size != N_MODULES:
        raise ValueError(f"se esperaban {N_MODULES} valores, llegaron {v.size}")
    return v.tobytes()


def ascii_preview(values: np.ndarray, layout: str = "4x4") -> str:
    """
    Vista rapida en consola. Util para depurar el mapeo sin hardware.

    layout="4x4"  retícula fisica (estrategia espejo)
    layout="5x3"  layout logico de la estrategia dedos: 5 columnas de dedo,
                  3 filas de nivel, mas el modulo 15 de apertura global.

    NOTA DE INTEGRACION: el orden fisico real de los 16 modulos de sOFIa lo
    define el equipo de mecatronica segun el cableado del bus. Si no coincide
    con el orden fila-mayor asumido aqui, no hay que tocar la logica: basta
    aplicar una tabla de permutacion de 16 posiciones justo antes de to_bytes().
    """
    ramp = " .:-=+*#%@"
    v = np.clip(np.asarray(values, dtype=float) / 255.0, 0, 1)
    rows = []
    if layout == "5x3":
        for r in range(3):
            rows.append(" ".join(ramp[int(x * 9)] * 2 for x in v[r * 5:(r + 1) * 5]))
        rows.append(f"apertura global (modulo 15): {ramp[int(v[15] * 9)] * 2}")
    else:
        for r in range(GRID):
            rows.append(" ".join(ramp[int(x * 9)] * 2 for x in v[r * GRID:(r + 1) * GRID]))
    return "\n".join(rows)
