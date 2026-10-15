"""
Pruebas offline: NO requieren camara ni mediapipe instalado.

Se genera una mano sintetica de 21 puntos con dedos parametrizados por flexion
y se verifica que la cadena completa (geometria -> features -> gestos -> matriz
-> trama) se comporte como debe. Esto permite avanzar el eje SC aunque la
camara o la ESP32 no esten disponibles, y sirve de base para las pruebas
automatizadas que pide la semana 12.

    python tests/test_offline.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import geometry as G                     # noqa: E402
from core import landmarks as LM                   # noqa: E402
from core import protocol                          # noqa: E402
from core.features import extract                  # noqa: E402
from core.gestures import GestureEngine, Giro, Saludo  # noqa: E402
from core.mapper import ModuleMapper, ascii_preview, to_bytes  # noqa: E402
from core.smoothing import OneEuroFilter, SlewLimiter          # noqa: E402


# ---------------------------------------------------------------------------
# Generador de mano sintetica
# ---------------------------------------------------------------------------
FINGER_X = {"pulgar": -0.55, "indice": -0.22, "medio": 0.0,
            "anular": 0.22, "menique": 0.44}
FINGER_LEN = {"pulgar": 0.75, "indice": 1.05, "medio": 1.15,
              "anular": 1.05, "menique": 0.85}


def synth_hand(curls: dict[str, float], scale: float = 120.0,
               center=(320.0, 240.0), rot_deg: float = 0.0,
               yaw_deg: float = 0.0) -> np.ndarray:
    """
    curls[dedo] en [0,1]: 0 = extendido, 1 = totalmente flexionado.
    Modelo simple: cada falange rota un angulo creciente respecto a la anterior.

    rot_deg  gira la mano en el plano de la imagen (inclinar la muneca)
    yaw_deg  gira la mano sobre su eje vertical: 0 = palma de frente,
             90 = mano de canto, 180 = dorso de frente. Es lo que permite
             simular el gesto GIRO sin camara.
    """
    pts = np.zeros((21, 3), dtype=float)
    pts[LM.WRIST] = (0.0, 0.0, 0.0)

    for finger, chain in LM.FINGER_CHAINS.items():
        c = float(np.clip(curls.get(finger, 0.0), 0.0, 1.0))
        x = FINGER_X[finger]
        seg = FINGER_LEN[finger] / 3.0
        base = np.array([x, 0.35 if finger != "pulgar" else 0.12, 0.0])
        pts[chain[0]] = base

        ang = 0.0
        pos = base.copy()
        for k in range(1, 4):
            # cada articulacion dobla mas: MCP poco, PIP y DIP mucho
            ang += c * np.radians(20 + 32 * k)
            d = np.array([np.sin(ang) * 0.12, np.cos(ang), -np.sin(ang) * 0.55])
            pos = pos + G.unit(d) * seg
            pts[chain[k]] = pos

    if yaw_deg:
        th = np.radians(yaw_deg)
        Ry = np.array([[np.cos(th), 0.0, np.sin(th)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(th), 0.0, np.cos(th)]])
        pts = pts @ Ry.T

    if rot_deg:
        th = np.radians(rot_deg)
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0],
                      [0, 0, 1.0]])
        pts = pts @ R.T

    pts = pts * scale
    pts[:, 0] += center[0]
    pts[:, 1] = center[1] - pts[:, 1]          # y de imagen crece hacia abajo
    return pts


OPEN = {f: 0.0 for f in LM.FINGER_NAMES}
FIST = {f: 1.0 for f in LM.FINGER_NAMES}
POINT = {**FIST, "indice": 0.0}
PEACE = {**FIST, "indice": 0.0, "medio": 0.0}


def feats(curls, ts=0, center=(320.0, 240.0), rot=0.0, yaw=0.0):
    p = synth_hand(curls, center=center, rot_deg=rot, yaw_deg=yaw)
    return extract(p, p, "Right", 0.99, ts)


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------
def test_extension():
    o, c, pt = feats(OPEN), feats(FIST), feats(POINT)
    print("  abierta:", {k: round(v, 2) for k, v in o.extensions.items()})
    print("  puno   :", {k: round(v, 2) for k, v in c.extensions.items()})
    print("  senala :", pt.code, "apertura", round(pt.aperture, 2))
    assert o.aperture > 0.75, o.aperture
    assert c.aperture < 0.30, c.aperture
    assert pt.fingers_up["indice"] and not pt.fingers_up["medio"]
    print("  OK extension de dedos")


def test_invarianza():
    """El gesto debe medirse igual sin importar escala, posicion o rotacion."""
    a = feats(OPEN)
    b = extract(synth_hand(OPEN, scale=260.0, center=(100.0, 380.0)),
                synth_hand(OPEN, scale=260.0, center=(100.0, 380.0)),
                "Right", 0.9, 0)
    c = feats(OPEN, rot=35.0)
    for name, x in (("escala+traslacion", b), ("rotacion 35deg", c)):
        d = abs(a.aperture - x.aperture)
        print(f"  desviacion apertura ({name}): {d:.4f}")
        assert d < 0.10, (name, d)
    print("  OK invarianza escala / traslacion / rotacion")


def test_canonico():
    p = synth_hand(OPEN, scale=200.0, center=(50.0, 450.0), rot_deg=40.0)
    q = synth_hand(OPEN, scale=90.0, center=(600.0, 120.0), rot_deg=-15.0)
    cp, cq = G.to_canonical(p), G.to_canonical(q)
    err = float(np.abs(cp - cq).max())
    print(f"  error maximo entre representaciones canonicas: {err:.4f}")
    assert err < 0.05, err
    print("  OK marco canonico")


def test_gestos():
    """Las 3 dinamicas y las 2 estaticas del catalogo final."""
    # PUNO sostenido
    eng = GestureEngine()
    got = None
    for i in range(40):
        ev = eng.update("p0_Right", feats(FIST, ts=i * 33))
        got = got or (ev.name if ev else None)
    print("  sostener puño ->", got)
    assert got == "PUNO", got

    # PAZ sostenida
    eng = GestureEngine()
    got = None
    for i in range(40):
        ev = eng.update("p0_Right", feats(PEACE, ts=i * 33))
        got = got or (ev.name if ev else None)
    print("  sostener V de paz ->", got)
    assert got == "PAZ", got

    # SALUDO: oscilacion lateral amplia con la mano abierta
    eng = GestureEngine()
    hit = None
    for i in range(80):
        x = 320.0 + 90.0 * np.sin(i * 0.45)
        ev = eng.update("p0_Right", feats(OPEN, ts=i * 33, center=(x, 240.0)))
        if ev and ev.name == "SALUDO":
            hit = ev.name
            break
    print("  oscilacion lateral amplia ->", hit)
    assert hit == "SALUDO"

    # PULSO: abrir -> cerrar -> abrir
    eng = GestureEngine()
    hit = None
    for i, cu in enumerate([OPEN] * 8 + [FIST] * 8 + [OPEN] * 8):
        ev = eng.update("p0_Right", feats(cu, ts=i * 33))
        if ev and ev.name == "PULSO":
            hit = ev.name
    print("  abrir-cerrar-abrir ->", hit)
    assert hit == "PULSO"
    print("  OK las 5 posturas")


def test_palma_quieta_no_es_saludo():
    """
    REGRESION. Una palma abierta sostenida NO debe leerse como SALUDO.

    Era el bug real: el detector contaba cambios de signo de la velocidad sin
    exigir amplitud por oscilacion, asi que el jitter de landmarks (2-3 px)
    bastaba para simular decenas de oscilaciones por segundo. Se simula aqui
    con ruido gaussiano encima de una mano inmovil.
    """
    rng = np.random.default_rng(11)
    for sigma_px, etiqueta in [(1.5, "jitter leve"), (4.0, "jitter fuerte")]:
        det = Saludo()
        disparos = 0
        for i in range(200):                       # ~6.6 s de mano quieta
            x = 320.0 + rng.normal(0, sigma_px)
            y = 240.0 + rng.normal(0, sigma_px)
            if det.update(feats(OPEN, ts=i * 33, center=(x, y)))[0]:
                disparos += 1
        print(f"  palma quieta con {etiqueta} ({sigma_px} px): "
              f"{disparos} saludos (debe ser 0)")
        assert disparos == 0

    # Una deriva LENTA de la mano tampoco es un saludo: va en una sola
    # direccion, no oscila.
    det = Saludo()
    disparos = sum(1 for i in range(120)
                   if det.update(feats(OPEN, ts=i * 33,
                                       center=(200.0 + i * 1.5, 240.0)))[0])
    print(f"  deriva lenta en linea recta: {disparos} saludos (debe ser 0)")
    assert disparos == 0
    print("  OK sin falsos saludos")


def test_separacion_posturas():
    """Cada postura debe activar SOLO su detector, con el motor completo."""
    casos = {
        "PUNO": [feats(FIST, ts=i * 33) for i in range(40)],
        "PAZ": [feats(PEACE, ts=i * 33) for i in range(40)],
        "PULSO": [feats(c, ts=i * 33)
                  for i, c in enumerate([OPEN] * 8 + [FIST] * 8 + [OPEN] * 8)],
        "SALUDO": [feats(OPEN, ts=i * 33,
                         center=(320.0 + 90.0 * np.sin(i * 0.45), 240.0))
                   for i in range(80)],
        "GIRO": _secuencia_giro(0.0, 180.0),
    }
    for esperado, secuencia in casos.items():
        eng = GestureEngine()
        vistos = []
        for f in secuencia:
            ev = eng.update("p0_Right", f)
            if ev:
                vistos.append(ev.name)
        unicos = sorted(set(vistos))
        print(f"  {esperado:7s} -> detectados: {unicos}")
        assert esperado in vistos, (esperado, vistos)
        assert unicos == [esperado], f"{esperado} activo tambien {unicos}"
    print("  OK las 5 posturas no se traslapan")


def _secuencia_giro(desde: float, hasta: float, quieto_ini=8, pasos=12,
                    quieto_fin=8, curls=None, dt=33):
    """Genera (features) de: mantener, voltear, mantener. Devuelve una lista."""
    curls = curls if curls is not None else OPEN
    yaws = ([desde] * quieto_ini
            + list(np.linspace(desde, hasta, pasos))
            + [hasta] * quieto_fin)
    return [feats(curls, ts=i * dt, yaw=float(y)) for i, y in enumerate(yaws)]


def test_giro():
    # --- 1. volteo palma -> dorso -------------------------------------------
    det = Giro()
    emitido, direccion = False, 0
    for f in _secuencia_giro(0.0, 180.0):
        ok, conf = det.update(f)
        if ok:
            emitido, direccion = True, det.direction
    print(f"  palma -> dorso: emitido={emitido} direccion={direccion}")
    assert emitido and direccion == -1

    # --- 2. volteo de regreso dorso -> palma --------------------------------
    det2 = Giro()
    emitido, direccion = False, 0
    for f in _secuencia_giro(180.0, 0.0):
        ok, _ = det2.update(f)
        if ok:
            emitido, direccion = True, det2.direction
    print(f"  dorso -> palma: emitido={emitido} direccion={direccion}")
    assert emitido and direccion == 1

    # --- 3. NO debe disparar con un salto de signo sin pasar por el canto ---
    # Simula que el clasificador confunde la mano por un frame: la normal se
    # invierte de golpe. Eso no es un giro fisico y hay que rechazarlo.
    det3 = Giro()
    salto = ([feats(OPEN, ts=i * 33, yaw=0.0) for i in range(10)]
             + [feats(OPEN, ts=(10 + i) * 33, yaw=180.0) for i in range(10)])
    falsos = sum(1 for f in salto if det3.update(f)[0])
    print(f"  salto de signo sin canto: {falsos} disparos (debe ser 0)")
    assert falsos == 0

    # --- 4. NO debe disparar con el puño cerrado ----------------------------
    det4 = Giro()
    falsos = sum(1 for f in _secuencia_giro(0.0, 180.0, curls=FIST)
                 if det4.update(f)[0])
    print(f"  puño girando: {falsos} disparos (debe ser 0)")
    assert falsos == 0

    # --- 5. una pausa larga antes del volteo NO lo invalida -----------------
    # El reloj mide la transicion, no la espera previa.
    det5 = Giro()
    emitido = False
    for f in _secuencia_giro(0.0, 180.0, quieto_ini=120):   # ~4 s de pausa
        if det5.update(f)[0]:
            emitido = True
    print(f"  pausa de 4 s antes del volteo: emitido={emitido}")
    assert emitido

    # --- 6. integrado en el motor completo ---------------------------------
    eng = GestureEngine()
    eventos = []
    for f in _secuencia_giro(0.0, 180.0):
        ev = eng.update("p0_Right", f)
        if ev:
            eventos.append((ev.name, ev.payload))
    nombres = [e[0] for e in eventos]
    print(f"  motor completo -> {eventos}")
    assert "GIRO" in nombres, nombres
    # Ninguna estatica debe colarse: la mano gira, no esta quieta.
    assert nombres == ["GIRO"], nombres
    payload = dict(eventos[nombres.index("GIRO")][1])
    assert payload["hacia"] == "DORSO", payload
    print("  OK gesto GIRO")


def test_mapeo():
    for strat in ModuleMapper.STRATEGIES:
        m = ModuleMapper(strat)
        v_open = m([feats(OPEN)])
        v_fist = m([feats(FIST)])
        assert v_open.shape == (16,) and len(to_bytes(v_open)) == 16
        assert v_open.sum() > v_fist.sum(), strat
        lay = "5x3" if strat == "dedos" else "4x4"
        print(f"  --- {strat}: mano abierta ---")
        print(ascii_preview(v_open, lay))
        print(f"  --- {strat}: puno ---")
        print(ascii_preview(v_fist, lay))
    print("  OK mapeo a 16 modulos")


def test_suavizado():
    """
    Se evalua el filtro en sus DOS regimenes, porque su valor esta justo en
    saber distinguirlos:

      (a) mano quieta -> debe matar el jitter (es donde se ve feo el temblor)
      (b) mano en movimiento rapido -> debe seguirla con retardo acotado

    Medir un solo numero global mezcla ruido con retardo y no dice nada.
    """
    rng = np.random.default_rng(7)

    # (a) reposo: senal constante contaminada con ruido de landmark
    f = OneEuroFilter(min_cutoff=1.5, beta=0.05)
    noisy, out = [], []
    for i in range(240):
        t = i / 60.0
        n = np.array([1.0]) + rng.normal(0, 0.05, 1)
        noisy.append(n[0]); out.append(f(n, t)[0])
    jitter_in = float(np.std(noisy[60:]))
    jitter_out = float(np.std(out[60:]))
    print(f"  reposo: jitter {jitter_in:.4f} -> {jitter_out:.4f} "
          f"({jitter_out / jitter_in:.0%} del original)")
    assert jitter_out < jitter_in * 0.45

    # (b) movimiento: escalon, se mide cuando alcanza el 90% del valor final
    f2 = OneEuroFilter(min_cutoff=1.5, beta=0.05)
    for i in range(60):
        f2(np.array([0.0]), i / 60.0)
    reach = None
    for i in range(60, 180):
        y = f2(np.array([1.0]), i / 60.0)[0]
        if reach is None and y >= 0.9:
            reach = (i - 60) * 1000 / 60
    print(f"  escalon: 90% del valor final en {reach:.0f} ms")
    assert reach is not None and reach < 400
    print("  OK One Euro (filtra en reposo, sigue en movimiento)")

    sl = SlewLimiter(size=16, max_rate=300.0, deadband=4.0)
    sl(np.zeros(16), 0.0)
    v = sl(np.full(16, 255.0), 0.033)
    print(f"  salto 0->255 limitado a {v[0]:.1f} en 33 ms")
    assert v[0] < 30.0
    print("  OK limitador de velocidad (protege los servos)")


def test_catalogo_coherente():
    """
    El motor y el protocolo deben declarar EXACTAMENTE las mismas posturas.
    Sin esta prueba, agregar un gesto y olvidar su codigo produce un fallo
    silencioso: se detecta en la PC y nunca llega a la ESP32.
    """
    del_motor = set(GestureEngine.PRIORITY)
    del_protocolo = set(protocol.GESTURE_CODES)
    print(f"  motor:      {sorted(del_motor)}")
    print(f"  protocolo:  {sorted(del_protocolo)}")
    assert del_motor == del_protocolo, del_motor ^ del_protocolo
    assert len(del_motor) == 5, f"se esperaban 5 posturas, hay {len(del_motor)}"

    codigos = list(protocol.GESTURE_CODES.values())
    assert len(set(codigos)) == len(codigos), "codigos de gesto duplicados"
    print("  OK 5 posturas, codigos unicos, motor y protocolo sincronizados")


def test_protocolo():
    payload = to_bytes(ModuleMapper("dedos")([feats(OPEN)]))
    frame = protocol.build(protocol.MsgType.MATRIX16, 42, payload)
    print(f"  trama MATRIX16: {len(frame)} bytes -> {frame[:8].hex(' ')} ...")
    f = protocol.parse(frame)
    assert f and f.type == protocol.MsgType.MATRIX16 and f.seq == 42
    assert f.payload == payload

    # resincronizacion: basura antes del sync
    assert protocol.parse(b"\x00\xff\x13" + frame) is not None

    # CRC corrupto debe rechazarse
    bad = bytearray(frame)
    bad[8] ^= 0xFF
    assert protocol.parse(bytes(bad)) is None
    print("  OK trama, resincronizacion y rechazo por CRC")


if __name__ == "__main__":
    tests = [
        ("Extension de dedos", test_extension),
        ("Invarianza", test_invarianza),
        ("Marco canonico", test_canonico),
        ("Gestos", test_gestos),
        ("Gesto GIRO", test_giro),
        ("Palma quieta no es saludo", test_palma_quieta_no_es_saludo),
        ("Separacion entre posturas", test_separacion_posturas),
        ("Mapeo 16 modulos", test_mapeo),
        ("Suavizado", test_suavizado),
        ("Catalogo coherente", test_catalogo_coherente),
        ("Protocolo", test_protocolo),
    ]
    for name, fn in tests:
        print(f"\n== {name} ==")
        fn()
    print("\nTodas las pruebas pasaron.")
