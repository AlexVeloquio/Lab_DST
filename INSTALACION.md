# INSTALACIÓN Y PRUEBAS — paso a paso

Guía para dejar el proyecto corriendo desde cero. Pensada para la configuración
**legacy** (`mp.solutions.hands`), que es la que pediste, pero el código soporta
las dos APIs y aquí se explican ambas.

---

## Paso 0 — Verificar la versión de Python

Esto es lo primero porque es lo que más gente traba.

```bash
python --version
```

| Tu versión | ¿Sirve para legacy? | Qué hacer |
|---|---|---|
| 3.8 – 3.11 | **Sí** | continúa al paso 1 |
| 3.12 | No (sí para tasks) | usa `requirements-tasks.txt`, o instala 3.11 |
| 3.13+ | No, para nada | **instala Python 3.11** |

`mediapipe==0.10.21` publica wheels hasta Python 3.11. En 3.12 `pip` no
encuentra nada y te va a decir "no matching distribution".

**Si necesitas instalar Python 3.11 en Windows:** descárgalo de
python.org/downloads, marca *"Add python.exe to PATH"* durante la instalación.
Puedes tener varias versiones conviviendo; para elegir cuál usar:

```bash
py -3.11 --version
```

---

## Paso 1 — Descomprimir y entrar a la carpeta

```bash
cd sofia_sc
```

Deberías ver `app.py`, `diagnostico.py`, `requirements.txt` y las carpetas
`core/`, `ui/`, `tests/`, `firmware/`, `models/`.

---

## Paso 2 — Crear el entorno virtual

**Windows (PowerShell o CMD):**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Sabes que funcionó porque el prompt cambia a `(.venv)`.

> Si PowerShell te bloquea el script de activación, corre una vez:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

---

## Paso 3 — Instalar dependencias

> **macOS con chip M1/M2/M3/M4:** usa `requirements.txt` (mediapipe 0.10.21).
> La rama 1.0.x **aborta el proceso** en Apple Silicon: inicializa Metal sin
> condición y falla con `Check failed: service_ Service is unavailable` en
> `DrishtiMetalHelper`. Es un `abort()` de C++, ni siquiera una excepción de
> Python. Pasar el delegado a CPU tampoco lo evita.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Instala `mediapipe==0.10.21`, `numpy<2` y `opencv-python==4.10.0.84`.
Son unos 80 MB, tarda un par de minutos.

**El `numpy<2` no es un capricho.** El wheel de mediapipe 0.10.21 declara
`Requires-Dist: numpy<2`. Si ya tenías numpy 2.x instalado globalmente y usas
el venv, no hay problema; pero si instalas fuera del venv, la importación truena
con errores de ABI que no dicen nada útil.

---

## Paso 4 — Diagnóstico

```bash
python diagnostico.py
```

Revisa Python, numpy, OpenCV, mediapipe, qué API trae, la cámara y la lógica
interna del proyecto. Si algo falla, te imprime el comando exacto para
arreglarlo. **No sigas si esto marca errores.**

Salida esperada, al final:

```
== MediaPipe ==
  version: 0.10.21
  [OK] API legacy (mp.solutions.hands): disponible
  [OK] el proyecto usara backend 'legacy' automaticamente

== Camara ==
  [OK] camara 0 funciona, resolucion 640x480

============================================================
Todo listo. Ya puedes correr:

    python app.py
```

---

## Paso 5 — Pruebas sin cámara ni hardware

```bash
python tests/test_offline.py
```

Corre 7 pruebas sobre una mano sintética: geometría, invarianza, marco canónico,
las 5 posturas, el mapeo a los 16 módulos, el suavizado y el protocolo. **No
necesita cámara, ni MediaPipe, ni la ESP32.** Termina con "Todas las pruebas
pasaron" y te imprime en ASCII cómo se vería la matriz 4×4 con mano abierta y
con puño.

Sirve para dos cosas: confirmar que la instalación de numpy está bien, y avanzar
en la lógica aunque no tengas el hardware a la mano.

---

## Paso 6 — La interfaz gráfica

La GUI es ahora el punto de entrada del proyecto:

```bash
python gui.py
```

Ventana con el video en vivo, el esqueleto, la postura detectada, las barras
de extensión por dedo, la previsualización de los 16 módulos y el panel de la
ESP32. Tres botones controlan el procesamiento:

| Botón | Qué le hace al hilo |
|---|---|
| **Iniciar** | Levanta el procesamiento y reinicia los detectores |
| **Detener** | El hilo se bloquea en una espera pasiva; no consume CPU |
| **Reanudar** | Continúa y limpia las ventanas temporales de los gestos dinámicos |

Tkinter viene en la biblioteca estándar, así que no hay nada que instalar. En
Linux, si falta, es `sudo apt install python3-tk`.

### Probar el enlace sin la ESP32

En una terminal levanta el simulador y en otra la GUI:

```bash
python esp32_simulador.py          # terminal 1
python gui.py --host 127.0.0.1     # terminal 2
```

Pulsa **Conectar** y luego **Ping**. Debe mostrar `RTT: 0.1 ms (5/5)`. El
simulador habla exactamente el mismo protocolo que el firmware, así que cuando
tengas la ESP32 real solo cambias `--host` por su IP.

Para probar la robustez con un enlace malo: `python esp32_simulador.py
--perdida 0.3` descarta el 30 % de las respuestas.

## Paso 6b — Corrida con la ventana de OpenCV

```bash
python app.py
```

Se abre una ventana con:
- el video espejado (te ves como en un espejo)
- el esqueleto de la mano, con cada dedo en su color; los dedos doblados se
  pintan apagados
- barras de extensión por dedo, abajo a la izquierda
- la previsualización de los 16 módulos como matriz 4×4
- arriba: FPS, manos detectadas, mapeo activo y estado de la ESP32

**Teclas:**

| Tecla | Acción |
|---|---|
| `q` | salir |
| `m` | alternar mapeo espejo ↔ dedos |
| `i` | mostrar/ocultar los índices de los 21 landmarks |
| `espacio` | pausar/reanudar el procesamiento |
| `s` | guardar captura PNG |
| `r` | reiniciar los filtros de suavizado |

`--cam` es `auto` por defecto: prueba los índices 0 a 3 y usa el primero que
entregue imagen. En varias Mac la cámara integrada **no es el índice 0**, y
abrir el equivocado da una ventana negra sin ningún mensaje de error. Puedes
forzarlo con `python app.py --cam 1`.

**Si la ventana no aparece o sale negra:** cierra Teams, Zoom o Meet, que se
quedan agarrando la cámara. **En macOS**, la terminal (o VS Code, o la app desde
donde lo corras) necesita permiso en *Ajustes → Privacidad y seguridad →
Cámara*; sin él, `VideoCapture` abre pero devuelve frames vacíos.

---

## Paso 7 — Probar las 5 posturas

Ponte a 1–2 m de la cámara, con la mano bien visible. La consola imprime
`[gesto] NOMBRE conf=0.87 ...` cada vez que detecta algo.

| Postura | Cómo provocarla |
|---|---|
| `PUNO` | puño bien cerrado, **quieto** ~0.4 s |
| `PAZ` | índice y medio arriba, anular y meñique recogidos, **quieta** ~0.4 s |
| `SALUDO` | mano abierta moviéndose de lado a lado con amplitud amplia, como diciendo adiós |
| `PULSO` | abrir → cerrar → abrir, rápido (menos de 1.4 s completo) |
| `GIRO` | mano abierta, voltearla de palma a dorso (o al revés) en menos de 1.5 s |

Para `GIRO` tienes un medidor en pantalla, abajo a la izquierda: una barra
`PALMA <-> DORSO` con una franja gris al centro. Esa franja es la **banda de
canto**, donde el signo de la normal no es confiable y el detector se niega a
comprometerse. Un giro válido cruza la franja completa. Si el marcador se queda
temblando dentro de la franja, estás girando demasiado despacio o la mano está
muy cerrada (se exige apertura > 0.40).

**Detalle importante:** las posturas estáticas (`PUNO` y `PAZ`) exigen que la
mano esté quieta, y "quieta" incluye **sin girar**. Es a propósito: una mano que
está volteando sigue teniendo, frame a frame, el mismo patrón de dedos, así que
sin ese filtro la estática dispararía a mitad del `GIRO`.

Cada postura vive en un eje de señal distinto — `SALUDO` en el desplazamiento
lateral, `PULSO` en la apertura, `GIRO` en la orientación de la palma, `PUNO` en
la apertura baja sostenida y `PAZ` en el patrón de dedos — así que no deberías
ver una leerse como otra. Si pasa, repórtalo: es un umbral mal calibrado para tu
cámara, no un límite del diseño.

Después de cada gesto hay un periodo refractario de 0.9 s durante el cual no se
detecta otro. También es a propósito: los servos tardan en ejecutar el patrón.

---

## Paso 8 — Ver los dos mapeos

Presiona `m` para alternar y observa la matriz 4×4 de la esquina:

- **espejo**: la matriz se comporta como una sombra de tu mano. Mano abierta
  ilumina ~14 celdas, señalar ~10, puño ~8.
- **dedos**: 5 columnas (una por dedo) × 3 niveles de flexión, más el módulo 15
  con la apertura global. No imita la forma, muestra el estado de cada dedo.

---

## Paso 9 — Conectar la ESP32 (cuando esté lista)

El proyecto soporta **dos transportes** con el mismo protocolo: WiFi/UDP y
Bluetooth Low Energy. La trama es idéntica en ambos; solo cambia el medio.

### Opción A — Bluetooth Low Energy

```bash
pip install bleak
python ble_ping.py --escanear     # ver qué hay cerca
python ble_ping.py                # conectar y hacer ping real
python gui.py --transporte ble    # la GUI por BLE
```

Sube `firmware/esp32_master_ble/` y necesitarás la librería **NimBLE-Arduino**.
La placa se anuncia como `sOFIa-Maestra`.

En macOS hay que dar permiso de Bluetooth a la terminal en *Ajustes →
Privacidad y seguridad → Bluetooth*, o el escaneo devuelve siempre vacío.

### Opción B — WiFi / UDP

1. Abre `firmware/esp32_master/esp32_master.ino` en Arduino IDE.
2. Cambia `SSID` y `PASS` por los de tu red (o configura la ESP32 como AP).
3. Instala la librería **Adafruit PWM Servo Driver** desde el gestor de
   librerías.
4. Sube el sketch y abre el monitor serie a 115200 baud. Anota la IP que
   imprime.
5. Corre:

```bash
python app.py --host 192.168.X.X --hz 25
```

El indicador de la esquina superior derecha pasa de `ESP32 --` a `ESP32 OK`
cuando el handshake responde. Si no conecta, la app sigue funcionando y
reintenta sola cada 500 ms.

**Failsafe:** si la ESP32 pasa 1.5 s sin recibir nada, manda todos los módulos a
reposo por su cuenta. Y `app.py` envía `STOP` explícito al cerrar.

---

## Paso 10 — Registrar métricas para el reporte

Lo más cómodo es la sesión guiada, que anuncia posturas en orden aleatorio y
anota qué pidió contra qué detectó:

```bash
python evaluacion.py --usuario U1 --iluminacion laboratorio --distancia 2.0 --repeticiones 20
python evaluacion.py --usuario U1 --reposo 3        # mide falsos positivos
python metricas.py "resultados/*.csv" --salida resultados/metricas.md
```

`metricas.py` produce las tablas en Markdown listas para pegar en el documento:
tasa de acierto por postura y usuario, por condición de captura, matriz de
confusión, tiempo de respuesta, FPS y falsos positivos.

El orden aleatorio no es un capricho: repetir veinte veces la misma postura
infla la tasa de acierto porque te calibras contra el detector, y además impide
medir confusiones entre posturas.

### Registro crudo durante el uso normal

```bash
python app.py --metricas resultados/pruebas_s6.csv
```

Genera un CSV con `t_ms, fps, manos, gesto, confianza, latencia_ms` por frame.
Es la base del entregable "Registro de métricas resultantes" de la semana 6.

Protocolo sugerido: ≥5 personas, distancias de 1 m / 2 m / 3.5 m, tres
condiciones de iluminación (laboratorio, contraluz de ventana, luz cálida baja),
20 repeticiones por postura. Mide tasa de acierto, **falsos positivos por minuto
en reposo** (la métrica que más molesta al público y la que más se olvida),
latencia y FPS promedio **y percentil 5** (el promedio esconde los tirones).

> El contraluz es el caso que va a fallar: las fotos del PDF muestran a sOFIa
> justo frente a un ventanal. Vale la pena medirlo temprano.

---

## Todas las opciones de `app.py`

```
--cam N|auto         índice de cámara, o auto para buscarla (default auto)
--ancho N            ancho de captura (default 640)
--alto N             alto de captura (default 480)
--manos N            máximo de manos a detectar (default 2)
--host IP            IP de la ESP32 Maestra
--puerto N           puerto UDP (default 4210)
--mapeo espejo|dedos estrategia de mapeo inicial
--pose               activar detección de cuerpo
--hz N               tasa de envío de MATRIX16 (default 25)
--metricas ARCHIVO   registrar CSV de métricas
--backend auto|legacy|tasks    forzar API de mediapipe
--complejidad 0|1    0 = modelo rápido, 1 = preciso (solo legacy)
```

---

## Problemas comunes

| Síntoma | Causa | Solución |
|---|---|---|
| `No matching distribution found for mediapipe==0.10.21` | Python 3.12+ | instala Python 3.11 |
| `module 'mediapipe' has no attribute 'solutions'` | tienes mediapipe 1.x | `pip install mediapipe==0.10.21 "numpy<2"` |
| Errores raros de ABI al importar mediapipe | numpy 2.x | `pip install "numpy<2"` |
| Ventana negra o no abre | otra app usa la cámara | cierra Teams/Zoom, o `--cam 1` |
| macOS: abre pero todo negro | falta permiso de cámara | Ajustes → Privacidad y seguridad → Cámara |
| `numpy 2.x incompatible` en el diagnóstico | falso positivo de versiones viejas del script | ya corregido; solo aplica a mediapipe 0.10.x |
| **macOS Apple Silicon: `Check failed: service_ Service is unavailable` + stack de `DrishtiMetalHelper`** | bug de mediapipe 1.0.x, aborta el proceso | `pip install "mediapipe==0.10.21" "numpy<2"` |
| `cv2` se comporta raro o versión inesperada | tienes `opencv-python` y `opencv-contrib-python` a la vez | `pip uninstall -y opencv-python opencv-contrib-python` y reinstala requirements |
| FPS por debajo de 15 | modelo pesado | `--complejidad 0` y/o `--manos 1` |
| Detecta gestos que no hiciste | umbrales flojos para tu setup | sube `up_threshold` en `features.py` o los umbrales en `gestures.py` |
| No detecta nada a contraluz | exposición de la cámara | iluminación frontal, o baja la exposición |
| Con `--pose` solo detecta una persona | limitación de `mp.solutions.pose` | usa `--backend tasks` (requiere mediapipe 1.x) |

---

## Si después quieres migrar a la API moderna

El código **ya la soporta**, no hay que reescribir nada:

```bash
pip install -r requirements-tasks.txt
python app.py --backend tasks
```

Descarga los modelos `.task` a `models/` la primera vez. La única diferencia
funcional a tu favor es que `--pose` sí detecta varias personas.

---

## Qué leer después

`README.md` tiene la investigación completa: cómo funciona MediaPipe Hands por
dentro, la comparación contra YOLO / OpenPose / OpenCV DNN, el diseño geométrico
de las posturas, las decisiones del mapeo a 16 módulos, el estándar de
comunicación y la tabla de riesgos del proyecto.
