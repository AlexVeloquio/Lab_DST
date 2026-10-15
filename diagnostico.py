"""
Diagnostico del entorno. Correr ANTES que app.py.

    python diagnostico.py

Verifica, en orden, todo lo que puede salir mal:
  1. version de Python
  2. numpy y su compatibilidad con mediapipe
  3. opencv
  4. mediapipe y que API trae (legacy / tasks)
  5. camara accesible
  6. logica interna (misma bateria que tests/test_offline.py)

Si algo falla, imprime el comando exacto para arreglarlo.
"""

from __future__ import annotations

import sys

OK = "  [OK] "
ER = "  [!!] "
WA = "  [..] "

problemas: list[str] = []


def seccion(titulo: str) -> None:
    print(f"\n== {titulo} ==")


# 1. Python -----------------------------------------------------------------
seccion("Python")
v = sys.version_info
print(f"  version: {sys.version.split()[0]}  ({sys.executable})")
if v.major == 3 and v.minor in (8, 9, 10, 11):
    print(OK + "compatible con mediapipe legacy (mp.solutions) y con tasks")
elif v.major == 3 and v.minor == 12:
    print(WA + "3.12 solo soporta la API de tasks; no hay 0.10.21 para esta version")
elif v.major == 3 and v.minor >= 13:
    print(ER + "no hay wheel oficial de mediapipe para 3.13+")
    problemas.append("Instala Python 3.11 y crea el venv con esa version.")
else:
    print(ER + "version no soportada")
    problemas.append("Usa Python 3.11.")

# 2. numpy ------------------------------------------------------------------
# Se evalua DESPUES de saber que mediapipe hay instalada: numpy 2.x solo es un
# problema para la rama 0.10.x. Con mediapipe 1.x es perfectamente correcto.
seccion("numpy")
numpy_major = None
try:
    import numpy as np
    numpy_major = int(np.__version__.split(".")[0])
    print(f"  version: {np.__version__}")
    print(OK + "importa correctamente "
          "(la compatibilidad se verifica en la seccion de MediaPipe)")
except ImportError:
    print(ER + "numpy no instalado")
    problemas.append("pip install numpy")

# 3. opencv -----------------------------------------------------------------
seccion("OpenCV")
try:
    import cv2
    print(f"  version: {cv2.__version__}")
    print(OK + "importa correctamente")
except ImportError:
    print(ER + "opencv no instalado")
    problemas.append("pip install opencv-python==4.10.0.84")

# 3b. cv2 duplicado ---------------------------------------------------------
# opencv-python y opencv-contrib-python entregan AMBOS el modulo cv2 y se pisan
# en el mismo directorio. pip no se queja, pero el cv2 que queda es impredecible.
# mediapipe 0.10.x instala opencv-contrib-python por su cuenta.
seccion("Paquetes de OpenCV instalados")
try:
    from importlib import metadata as _md
    instalados = []
    for nombre in ("opencv-python", "opencv-contrib-python",
                   "opencv-python-headless", "opencv-contrib-python-headless"):
        try:
            instalados.append(f"{nombre}=={_md.version(nombre)}")
        except _md.PackageNotFoundError:
            pass
    if not instalados:
        print(WA + "ninguno detectado por metadata (puede ser normal)")
    else:
        for i in instalados:
            print(f"  {i}")
        if len(instalados) > 1:
            print(ER + "hay mas de un paquete que entrega cv2: se pisan entre si")
            problemas.append(
                "pip uninstall -y opencv-python opencv-contrib-python && "
                "pip install opencv-contrib-python")
        else:
            print(OK + "un solo proveedor de cv2")
except Exception as exc:
    print(WA + f"no se pudo revisar: {exc}")

# 4. mediapipe --------------------------------------------------------------
seccion("MediaPipe")
try:
    import mediapipe as mp
    print(f"  version: {getattr(mp, '__version__', 'desconocida')}")

    legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    print((OK if legacy else WA) +
          f"API legacy (mp.solutions.hands): {'disponible' if legacy else 'NO disponible'}")

    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        tasks = True
    except Exception:
        tasks = False
    print((OK if tasks else WA) +
          f"API tasks (HandLandmarker): {'disponible' if tasks else 'NO disponible'}")

    # Bug conocido: 1.0.x aborta en Mac con Apple Silicon (Metal no disponible)
    import platform as _plat
    mac_arm = sys.platform == "darwin" and _plat.machine() in ("arm64", "aarch64")
    mp_mayor = str(getattr(mp, "__version__", "0")).split(".")[0]
    if mac_arm and mp_mayor >= "1":
        print(ER + "mediapipe 1.0.x ABORTA en Mac con Apple Silicon "
                   "(DrishtiMetalHelper / Metal service unavailable)")
        print(ER + "no es atrapable desde Python: mata el proceso completo")
        problemas.append(
            'pip uninstall -y mediapipe opencv-python opencv-contrib-python && '
            'pip install "mediapipe==0.10.21" "numpy<2"')

    if not legacy and not tasks:
        problemas.append("Reinstala mediapipe: pip install mediapipe==0.10.21")
    elif legacy:
        print(OK + "el proyecto usara backend 'legacy' automaticamente")
    else:
        print(OK + "el proyecto usara backend 'tasks' "
                   "(descarga los modelos .task la primera vez)")
        print(OK + "con este backend, --pose SI detecta varias personas")

    # Veredicto de numpy, ahora si con contexto
    mp_major = str(getattr(mp, "__version__", "0")).split(".")[0]
    if numpy_major is not None:
        if mp_major == "0" and numpy_major >= 2:
            print(ER + f"numpy {numpy_major}.x con mediapipe 0.10.x: "
                       "esta rama exige numpy<2")
            problemas.append('pip install "numpy<2"')
        else:
            print(OK + f"numpy {numpy_major}.x es compatible con esta mediapipe")

except ImportError as exc:
    print(ER + f"no se pudo importar: {exc}")
    problemas.append('pip install mediapipe==0.10.21 "numpy<2"')

# 5. camara -----------------------------------------------------------------
seccion("Camara")
try:
    import cv2
    buenas = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(OK + f"camara {idx} funciona, resolucion "
                           f"{frame.shape[1]}x{frame.shape[0]}")
                buenas.append(idx)
        cap.release()

    if not buenas:
        print(ER + "ninguna camara respondio en los indices 0-3")
        problemas.append("Cierra Teams/Zoom/Meet y vuelve a intentar. "
                         "En macOS, da permiso de camara a la terminal en "
                         "Ajustes > Privacidad y seguridad > Camara.")
    elif 0 not in buenas:
        print(WA + f"la camara 0 NO responde; la util es la {buenas[0]}")
        print(WA + "app.py la detecta sola (--cam auto), pero puedes forzarla: "
                   f"python app.py --cam {buenas[0]}")
except Exception as exc:
    print(ER + f"error al abrir la camara: {exc}")

# 6. logica interna ---------------------------------------------------------
seccion("Logica del proyecto (sin camara)")
try:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
    from test_offline import (                                  # noqa: E402
        test_extension, test_invarianza, test_canonico, test_gestos,
        test_mapeo, test_suavizado, test_protocolo,
    )
    import io
    import contextlib

    for nombre, fn in [
        ("extension de dedos", test_extension),
        ("invarianza", test_invarianza),
        ("marco canonico", test_canonico),
        ("gestos", test_gestos),
        ("mapeo 16 modulos", test_mapeo),
        ("suavizado", test_suavizado),
        ("protocolo", test_protocolo),
    ]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            print(OK + nombre)
        except Exception as exc:
            print(ER + f"{nombre}: {exc}")
            problemas.append(f"Falla interna en {nombre}. Reporta el error.")
except Exception as exc:
    print(ER + f"no se pudieron correr las pruebas: {exc}")

# Resumen -------------------------------------------------------------------
print("\n" + "=" * 60)
if problemas:
    print("HAY QUE ARREGLAR ESTO ANTES DE CORRER app.py:\n")
    for i, p in enumerate(problemas, 1):
        print(f"  {i}. {p}")
    sys.exit(1)
else:
    print("Todo listo. Ya puedes correr:\n\n    python app.py\n")
