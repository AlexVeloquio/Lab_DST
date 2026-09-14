# sOFIa — Eje Técnico SC
### Detección de manos y cuerpo con MediaPipe + OpenCV, y control de los 16 módulos

Investigación técnica y código base para el Laboratorio de Desarrollo de Soluciones
Tecnológicas (ITESO, DESI). Cubre el análisis de librerías, el diseño geométrico de
las posturas, el mapeo mano → 16 módulos y el estándar de comunicación PC ↔ ESP32.

---

## 0. Lo primero, porque invalida casi todos los tutoriales

**`mediapipe.solutions` ya no existe.**

El código que aparece en el 99% de los tutoriales, videos y ejemplos de curso:

```python
import mediapipe as mp
hands = mp.solutions.hands.Hands(max_num_hands=2)      # AttributeError
mp.solutions.drawing_utils.draw_landmarks(...)         # AttributeError
```

El módulo legacy fue retirado a partir de `mediapipe 0.10.30` (diciembre 2025) y
está definitivamente ausente en la rama 1.x. La versión publicada en PyPI al
momento de escribir esto es **1.0.1 (14 de agosto de 2026)**, con wheels para
Python 3.9–3.12 en Windows x86-64. **No hay wheel oficial para Python 3.13.**

Esto deja dos caminos, y conviene elegirlo de forma explícita y documentada
porque es exactamente el tipo de decisión que se defiende en la semana 3:

| | Opción A: congelar `mediapipe==0.10.21` | Opción B: API de Tasks (recomendada) |
|---|---|---|
| API | `mp.solutions.hands` legacy | `vision.HandLandmarker` |
| Tutoriales | funcionan tal cual | hay que traducirlos |
| numpy | obligado a `<2` | sin restricción |
| Modo asíncrono | no | sí (`LIVE_STREAM`) |
| Mantenimiento | rama muerta | rama vigente |
| Riesgo a diciembre | el entregable nace obsoleto | ninguno |

**Este proyecto soporta las dos.** `core/tracker.py` detecta sola cuál API está
instalada y se adapta; el resto del código no se entera. Se puede forzar con
`--backend legacy` o `--backend tasks`.

La configuración por defecto de `requirements.txt` es la **legacy**
(`mediapipe==0.10.21` + `numpy<2` + Python 3.11), que es la que conserva
`mp.solutions`. Para la moderna está `requirements-tasks.txt`.

Dos cosas que cambian según el backend, y conviene tenerlas presentes:

- **La API legacy es síncrona.** `hands.process()` bloquea. Si se llamara desde
  el hilo de la GUI, la interfaz se congelaría cada vez que la inferencia se
  atrase — justo lo que prohíbe la semana 9. Por eso el backend legacy corre en
  un hilo trabajador propio que procesa siempre el frame más nuevo y descarta
  los viejos. La API de Tasks ya trae eso resuelto con `detect_async()`.
- **`mp.solutions.pose` es de UNA sola persona.** No tiene equivalente a
  `num_poses`. Si el eje SC va a demostrar detección en varias personas
  (requisito 1), ahí sí hace falta el backend tasks. La detección de varias
  manos funciona en los dos.

> La diapositiva 8 del PDF del curso lista `Ultralytics 8.3.24` y no menciona
> MediaPipe. Si el equipo va a usar MediaPipe, vale la pena avisarle al profesor
> y dejar la justificación por escrito en el reporte.

---

## 1. Qué hace realmente MediaPipe Hands

### 1.1 Arquitectura de dos etapas

MediaPipe no corre un detector sobre toda la imagen en cada frame. Usa una
cascada:

1. **Palm detector** (BlazePalm, tipo SSD). Busca *palmas*, no manos completas.
   Detectar palmas es mucho más fácil: son rígidas, cuadradas y sin dedos
   articulados que disparen la varianza. Devuelve una caja orientada.
2. **Hand landmark model.** Sobre el recorte de esa caja, hace regresión directa
   de los 21 puntos, incluso los ocluidos (los infiere por contexto anatómico).
3. **Tracking.** Mientras la confianza de presencia se mantenga, los frames
   siguientes **saltan la etapa 1** y reusan la caja del frame anterior.

De ahí viene el rendimiento: el detector pesado corre esporádicamente, no en
cada frame. Y de ahí también viene la implicación práctica más importante:

> Bajar `min_tracking_confidence` hace que la mano se pierda menos, pero
> prolonga el uso de una caja vieja y los landmarks se degradan. Bajar
> `min_hand_detection_confidence` hace lo contrario: redetecta más seguido,
> más costoso pero más fresco. **No son intercambiables** aunque los dos
> "mejoren la detección".

### 1.2 Los 21 landmarks

```
            8   12  16  20        4 = punta pulgar
            |   |   |   |         8 = punta índice
            7  11  15  19        12 = punta medio
            |   |   |   |        16 = punta anular
        4   6  10  14  18        20 = punta meñique
       /    |   |   |   |
      3     5---9--13--17        5,9,13,17 = nudillos (MCP)
     /       \  |  /  /          0 = muñeca
    2         \ | /  /
     \         \|/  /
      1---------0--'
```

Cada dedo sigue el patrón **MCP → PIP → DIP → punta** (el pulgar:
CMC → MCP → IP → punta). Esa regularidad es la que permite escribir una sola
función de flexión que sirve para los cinco dedos (`core/features.py`).

### 1.3 Tres sistemas de coordenadas (la fuente de errores más común)

| Salida | Rango | Origen | Para qué sirve |
|---|---|---|---|
| `hand_landmarks` | x,y ∈ [0,1] normalizados a la imagen; z relativo a la muñeca | esquina de la imagen | **dibujar** en pantalla |
| `hand_world_landmarks` | metros aproximados | centro geométrico de la mano | **medir ángulos y distancias** |
| `handedness` | categoría + score | — | saber si es izquierda o derecha |

El error clásico es calcular ángulos con las coordenadas normalizadas. Como x
está normalizado al **ancho** y y al **alto**, en un video 16:9 un ángulo de 45°
se mide como 28°. Y además la perspectiva deforma todo con la distancia. Los
ángulos van **siempre** sobre `world_landmarks`; la pantalla solo para pintar.

`handedness` se reporta respecto a la imagen **sin espejar**. Si haces
`cv2.flip()` para que el usuario se vea como en un espejo (recomendable,
`app.py` lo hace), izquierda y derecha te llegan cambiadas. Hay que decidir la
convención una vez y anotarla.

---

## 2. Elección de librería: por qué MediaPipe y no las otras

El PDF del curso propone cuatro alternativas. Comparación centrada en lo que
este proyecto necesita:

| Criterio | **MediaPipe Hands** | YOLO11-pose | OpenPose | OpenCV DNN + TF |
|---|---|---|---|---|
| Puntos de la mano | **21 por mano, con z** | 0 (solo 17 de cuerpo) | 21 (con modelo de mano) | según modelo |
| Puntos de cuerpo | 33 (Pose) | 17 | 25 | 17–18 |
| CPU sin GPU | **30–60 FPS** | 10–25 FPS | <5 FPS | 5–15 FPS |
| Multi-persona | sí (Pose) | **sí, nativo con ID** | sí | limitado |
| Instalación en Windows | `pip install` | `pip install` | **compilar con CUDA** | manual |
| Licencia | Apache 2.0 | **AGPL-3.0** | no comercial | Apache/BSD |

**Conclusión para el Eje SC:** YOLO-pose **no detecta dedos**. Sus 17 keypoints
COCO terminan en la muñeca. Si el proyecto exige "leer todos los componentes de
la mano", YOLO queda descartado para esa tarea sin importar qué tan bueno sea
detectando personas. OpenPose sí da dedos, pero exige compilar con CUDA y no
alcanza tiempo real en CPU: inviable para una PC de laboratorio.

**Arquitectura adoptada — los dos modelos en paralelo:**

```
                 ┌──► HandLandmarker  (21×N pts) ──┐
   frame RGB ────┤                                  ├──► asociación ──► features
                 └──► PoseLandmarker  (33×M pts) ──┘
```

`HandLandmarker` entrega una **lista plana de manos sin identidad de persona**:
con dos personas frente a sOFIa no dice de quién es cada mano. El requisito 1
pide detectar posturas en varias personas, así que hace falta `PoseLandmarker`
para obtener el esqueleto y asociar cada mano a su dueño comparando la muñeca
de la mano con las muñecas del cuerpo (landmarks 15 y 16). Eso está resuelto en
`associate_hands_to_poses()`.

> Sobre la licencia: Ultralytics es **AGPL-3.0**. Para un proyecto académico es
> irrelevante, pero si sOFIa se exhibe públicamente o el código se publica,
> AGPL obliga a liberar todo el proyecto bajo la misma licencia. MediaPipe es
> Apache 2.0 y no impone nada. Vale la pena mencionarlo en el reporte.

---

## 3. De 21 puntos a un gesto: reglas geométricas

MediaPipe entrega **puntos, no gestos**. Todo el valor de ingeniería del eje SC
está en la conversión. Hay dos caminos:

**(a) Reglas geométricas explícitas** — el que pide el cronograma ("distancias
entre puntos detectados, ángulos entre articulaciones"). Sin dataset, sin
entrenamiento, depurable, y cada umbral se puede justificar y graficar en el
reporte.

**(b) Clasificador entrenado** sobre los 21 puntos normalizados (un MLP pequeño
o un KNN). Mejor para gestos sutiles, pero exige grabar y etiquetar datos.
*Si el equipo va por aquí, la entrada correcta del clasificador son las
coordenadas canónicas de `geometry.to_canonical()`, nunca los píxeles crudos.*

Este proyecto implementa **(a)**, dejando (b) preparado.

### 3.1 Invarianza: el requisito que nadie escribe pero todos necesitan

Un gesto debe medirse igual a 1 m que a 4 m de la cámara, con la muñeca recta o
inclinada, con mano izquierda o derecha. Tres herramientas, en `geometry.py`:

- **Normalizar por el tamaño de palma.** Toda distancia se divide entre
  `‖MCP_medio − muñeca‖`. Las magnitudes quedan en "palmas", adimensionales.
  Un umbral de 1.1 palmas significa lo mismo a cualquier distancia.
- **Marco canónico.** Base ortonormal anclada a la mano: origen en la muñeca,
  eje Y hacia los dedos, eje X a lo ancho de la palma, eje Z normal. Proyectar
  ahí elimina traslación, rotación y escala de un solo golpe.
- **Espejar la izquierda.** Se espeja en X y se evalúa con las mismas reglas de
  la derecha. Un solo conjunto de umbrales para las dos manos.

### 3.2 Extensión de dedos: dos evidencias, no una

`features.finger_extension()` devuelve un escalar continuo en [0,1] por dedo
combinando **dos medidas que fallan en casos distintos**:

- **Ángulos de flexión** (PIP y DIP). Robustos ante escala, pero degeneran en
  escorzo: cuando el dedo apunta hacia la cámara los tres puntos se proyectan
  casi encima y el ángulo se vuelve ruido puro.
- **Razón de distancias** `d(punta,muñeca) / d(MCP,muñeca)`. Sobrevive al
  escorzo, pero se confunde cuando la mano se pone de perfil.

Fusión 65/35 a favor del ángulo, porque el caso dominante frente a sOFIa es
mano de frente. El resultado es **continuo**, no binario: sirve tanto para
decidir "dedo arriba" con un umbral como para alimentar directamente el nivel de
un servo.

El pulgar necesita su propio rango (nunca llega a 180°); usar el mismo umbral
que los demás es el bug más común en este tipo de código.

---

## 4. Las 5 posturas

El cronograma exige 5 posturas con **al menos 3 dinámicas**. Diseñadas como
"conversación" visual con sOFIa:

| Postura | Tipo | Detección | Reacción sugerida de sOFIa |
|---|---|---|---|
| `PALMA_ABIERTA` | estática | apertura > 0.72, ≥4 dedos arriba, **y mano quieta** | despliega el panel completo |
| `PUNO` | estática | apertura < 0.28, ≤1 dedo arriba, **y mano quieta** | se contrae al centro |
| `SALUDO` | dinámica | ≥3 inversiones de velocidad horizontal en 1.2 s, amplitud ≥1.1 palmas | devuelve una ola |
| `BARRIDO` | dinámica | desplazamiento neto ≥1.6 palmas con monotonía >0.8 | onda viajera en la dirección |
| `PULSO` | dinámica | abrir → cerrar → abrir en <1.1 s (máquina de 3 estados) | latido de toda la matriz |

### Por qué "y mano quieta" no es un detalle

Un **saludo también es, frame a frame, una mano abierta**. Sin el requisito de
quietud, `PALMA_ABIERTA` dispara a los 350 ms y el saludo nunca alcanza a
completar su ventana de 1.2 s: el gesto dinámico es literalmente inalcanzable.
Esto salió en las pruebas automatizadas (`tests/test_offline.py`) y es el tipo
de hallazgo que conviene documentar.

La regla que resuelve el traslape: **quieto = estático, en movimiento =
dinámico**. Medido en palmas, no en píxeles.

Distinguir `SALUDO` de `BARRIDO` usa la misma señal con criterios opuestos:
saludo = muchas inversiones; barrido = cero inversiones (monotonía alta). Son
la misma familia de señal separada por una métrica de forma.

---

## 5. Suavizado: tres problemas distintos, tres herramientas

Confundirlos es el error típico. Requisito explícito de la semana 6.

**1. Jitter de landmarks** (el punto vibra 2–3 px con la mano quieta)
→ **filtro One Euro**. Un promedio móvil de N frames introduce siempre N/2
frames de retardo — se siente como que la mano va adelantada del esqueleto. One
Euro adapta su frecuencia de corte a la velocidad: filtra fuerte en reposo y
deja pasar en movimiento rápido.

Medido en `tests/test_offline.py`:
```
reposo:   jitter 0.0451 → 0.0119   (26% del original)
escalón:  90% del valor final en 233 ms
```

**2. Parpadeo de clasificación** (PUÑO/PALMA/PUÑO en frames consecutivos)
→ **voto por mayoría** en ventana de 5–7 frames + **histéresis**: entrar cuesta
más que salir (0.68 para activar, 0.55 para desactivar), lo que elimina el
rebote cuando la señal se queda justo sobre el umbral.

**3. Chattering mecánico** (el servo tiembla persiguiendo microcambios)
→ **banda muerta + limitador de velocidad**. Este es el que más importa: sOFIa
es hardware. Sin esto, un landmark ruidoso se traduce en servos persiguiendo
ruido a 30 Hz — ruido audible, calentamiento, desgaste y tráfico inútil hacia
la ESP32. Medido: un salto 0→255 queda limitado a ~10 unidades en 33 ms.

Hay una **segunda rampa dentro del firmware de la ESP32**, a propósito: la PC
puede reiniciarse o enviar basura, y el hardware no debería depender de que el
software se comporte.

---

## 6. Mapeo mano → 16 módulos

MediaPipe da 21 puntos; sOFIa tiene 16 módulos. **No hay correspondencia
natural: hay que diseñarla.** Se implementan dos estrategias porque responden a
intenciones distintas, y ambas son defendibles ante el jurado.

### ESPEJO — retícula 4×4 espacial

La pared se comporta como una sombra de la mano. Se densifica el esqueleto
(se interpolan puntos a lo largo de cada hueso — con solo 21 puntos las celdas
quedan huecas y el reflejo se ve punteado) y se rasteriza sobre la retícula con
un núcleo gaussiano.

Dos decisiones que cambian por completo el resultado:

- **Escalar por tamaño de palma, no por caja delimitadora.** Si se normaliza
  por la caja, el puño se estira hasta llenar la retícula igual que la mano
  abierta y ambos gestos se ven idénticos en la pared. Con la palma como
  unidad, la mano abierta ocupa los 4×4 y el puño se concentra al centro.
- **Cobertura, no densidad.** Con densidad cruda el puño (muchos puntos
  apretados en pocas celdas) ilumina *más* que la mano abierta — exactamente lo
  contrario de lo que el público espera. La saturación exponencial convierte la
  medida en "esta celda está ocupada por la mano".

Contraste medido con `sigma=0.38` (celdas por encima del 30%):
mano abierta 14 · señalar 10 · puño 8.

### DEDOS — 5×3 + 1 semántico

15 módulos = 5 dedos × 3 niveles de flexión (una barra de progreso por dedo) +
1 módulo de apertura global. No imita la forma, comunica el **estado de cada
dedo**. Es el mapeo correcto cuando lo que importa es que se lea qué dedo se
movió. La transición es continua, no binaria, para que los servos se muevan
suave.

> **Nota de integración:** el código asume orden fila-mayor de los 16 módulos.
> Si el cableado real del bus usa otro orden, no hay que tocar la lógica: basta
> una tabla de permutación de 16 posiciones justo antes de `to_bytes()`.

Con dos manos en escena se toma el **máximo** por módulo, no el promedio: el
promedio apaga la mano más expresiva y el resultado se ve muerto.

---

## 7. Estándar de comunicación PC ↔ ESP32

Entregable formal de la semana 9. Implementado en `core/protocol.py` y
`firmware/esp32_master/esp32_master.ino`.

### Transporte: UDP para el flujo, ACK para los eventos

- El flujo `MATRIX16` es **de estado, no de eventos**: si se pierde una trama,
  la siguiente (33 ms después) ya trae el estado correcto. TCP sería peor —
  reordenaría y entregaría tarde un dato ya obsoleto, produciendo el efecto
  "el panel va retrasado y luego se acelera".
- Los comandos de evento (`GESTURE`, `STOP`) sí importan: número de secuencia,
  ACK y hasta 3 reintentos.

### Formato de trama (24 bytes para MATRIX16)

```
 0      1      2      3      4      5         5+N    5+N+1
+------+------+------+------+------+------+---------+-------+
| 0xA5 | 0x5A | VER  | TYPE | SEQ  | LEN  | PAYLOAD | CRC16 |
+------+------+------+------+------+------+---------+-------+
  sync   sync    1B     1B     1B     1B    N bytes    2B
```

El **doble byte de sincronía** permite resincronizar el parser de la ESP32 si
se pierde un byte, sin reiniciar el enlace. `CRC-16/CCITT-FALSE` sobre
VER..PAYLOAD.

| TYPE | Mensaje | Dirección | ACK |
|---|---|---|---|
| 0x01 | `HELLO` | PC → ESP32 | sí |
| 0x02 | `ACK` | ESP32 → PC | — |
| 0x03 | `NACK` (+ código de error) | ESP32 → PC | — |
| 0x10 | `GESTURE` (código, confianza, extra) | PC → ESP32 | sí |
| 0x11 | `MATRIX16` (16 bytes) | PC → ESP32 | no |
| 0x20 | `HEARTBEAT` (cada 500 ms) | PC → ESP32 | no |
| 0x21 | `STATUS` (telemetría) | ESP32 → PC | — |
| 0x40 | `STOP` (paro seguro) | PC → ESP32 | sí |

**Failsafe:** si la ESP32 pasa 1.5 s sin recibir nada, manda todos los módulos a
reposo por su cuenta. Cubre el caso de que la PC se cuelgue, se cierre la
aplicación o se caiga el WiFi — los servos no se quedan trabados en una
posición forzada. `app.py` además envía `STOP` explícito al cerrar.

**Ancho de banda:** 24 bytes × 25 Hz = 600 B/s. Irrelevante. La restricción real
es la **latencia**, no el throughput: WiFi doméstico añade 5–15 ms típicos con
picos de 100 ms+ si el canal está saturado. Vale la pena medirlo con el CSV de
métricas y, si hace falta, poner la ESP32 en modo AP para tener el canal sola.

---

## 8. Cómo correrlo

Guía detallada en **`INSTALACION.md`**. Versión corta:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Verificar entorno, cámara y lógica antes de nada
python diagnostico.py

# Pruebas sin cámara ni hardware (útiles desde el primer día)
python tests/test_offline.py

# Solo visión
python app.py

# Con la ESP32
python app.py --host 192.168.4.1 --mapeo espejo --hz 25

# Multi-persona + registro de métricas
python app.py --pose --metricas resultados/pruebas_s6.csv

# Forzar una API concreta
python app.py --backend legacy      # mp.solutions
python app.py --backend tasks       # HandLandmarker
```

Con backend tasks, el modelo `.task` se descarga solo la primera vez a
`models/`. Con backend legacy no hay descarga: los modelos vienen dentro del
paquete.

Teclas: `q` salir · `m` alternar mapeo · `i` ids de landmarks · `espacio` pausa
· `s` captura · `r` reiniciar filtros.

---

## 9. Estructura

```
sofia_sc/
├── app.py                  loop principal, hilos, teclado, métricas CSV
├── diagnostico.py          verifica entorno, cámara y lógica antes de correr
├── INSTALACION.md          paso a paso de instalación y pruebas
├── requirements.txt        configuración legacy (mp.solutions, Python 3.11)
├── requirements-tasks.txt  configuración moderna (mediapipe 1.x)
├── core/
│   ├── landmarks.py        constantes y topología (reemplaza HAND_CONNECTIONS)
│   ├── geometry.py         marco canónico, ángulos, normalización  [sin deps]
│   ├── features.py         extensión de dedos, apertura, pinza      [sin deps]
│   ├── smoothing.py        One Euro, voto, histéresis, slew limiter [sin deps]
│   ├── gestures.py         las 5 posturas + coordinador             [sin deps]
│   ├── mapper.py           mano → 16 módulos, 2 estrategias         [sin deps]
│   ├── protocol.py         trama, CRC16, cliente UDP                [sin deps]
│   └── tracker.py          MediaPipe, ambos backends (único que lo importa)
├── ui/overlay.py           dibujo OpenCV (reemplaza drawing_utils)
├── tests/test_offline.py   mano sintética, 7 pruebas, sin cámara
└── firmware/esp32_master/  receptor UDP → PCA9685
```

**`[sin deps]` significa que ese archivo solo necesita numpy.** Es deliberado:
toda la lógica de negocio se prueba sin cámara, sin MediaPipe y sin hardware.
Eso desbloquea trabajar en paralelo con el equipo de mecatrónica y es lo que
permite tener pruebas automatizadas desde la semana 3, en vez de esperar a que
sOFIa esté armada.

### Hilos

```
hilo cámara     lee sin parar, guarda SOLO el frame más nuevo
                (descartar frames viejos > acumular retardo)
hilo MediaPipe  interno de la librería, entrega por callback
hilo principal  dibuja, decide, envía
```

Los timestamps de `LIVE_STREAM` deben ser **estrictamente crecientes en ms** o
MediaPipe lanza excepción. Por eso se usa un contador monótono propio y no
`time.time()`.

---

## 10. Métricas y pruebas (semanas 6 y 12)

`app.py --metricas archivo.csv` registra por frame: `t_ms, fps, manos, gesto,
confianza, latencia_ms`.

Protocolo de prueba sugerido:

| Variable | Niveles |
|---|---|
| Usuarios | ≥5 personas, distinto tono de piel y tamaño de mano |
| Distancia | 1 m, 2 m, 3.5 m |
| Iluminación | luz de laboratorio, contraluz de ventana, luz cálida baja |
| Fondo | pared lisa, fondo con personas caminando |
| Repeticiones | 20 por postura por condición |

Métricas: **tasa de acierto** (detectados / intentados), **falsos positivos por
minuto** en reposo — la métrica que más molesta al público y la que más se
olvida medir —, **latencia gesto→movimiento** (con cronómetro en video a 240 fps
del celular), y **FPS promedio y percentil 5** (el promedio esconde los tirones).

**Contraluz de ventana es el caso que va a fallar.** Las fotos del PDF muestran
sOFIa justo frente a un ventanal. Conviene medirlo temprano y, si hace falta,
pedir iluminación frontal o bajar la exposición de la cámara por software.

---

## 11. Riesgos identificados

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Python 3.13 sin wheel de MediaPipe | bloqueante | fijar 3.11 o 3.12 desde el día 1 |
| Copiar tutoriales con `mp.solutions` | horas perdidas | este README, sección 0 |
| Contraluz del ventanal | detección irregular | medir temprano; iluminación frontal |
| Orden físico de los 16 módulos ≠ fila-mayor | mapeo espejado o revuelto | tabla de permutación antes de `to_bytes()` |
| Latencia WiFi variable | reacción a destiempo | ESP32 en modo AP; medir con el CSV |
| Traslape saludo / palma abierta | gesto dinámico inalcanzable | requisito de quietud (§4) |
| Chattering de servos | desgaste, ruido | slew limiter en PC **y** rampa en firmware |
| AGPL de Ultralytics si se usa YOLO | contamina la licencia del proyecto | quedarse en MediaPipe (Apache 2.0) |

---

## 12. Fuentes

- Guía oficial de Hand Landmarker para Python —
  https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- Guía oficial de Pose Landmarker —
  https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
- `mediapipe` en PyPI (1.0.1, 14-ago-2026; clasificadores Python 3.9–3.12) —
  https://pypi.org/project/mediapipe/
- Issue #6192, ausencia de `mediapipe.solutions` desde 0.10.31 —
  https://github.com/google-ai-edge/mediapipe/issues/6192
- Issue #6159, falta de soporte para Python 3.13 —
  https://github.com/google-ai-edge/mediapipe/issues/6159
- Ejemplos oficiales en Python —
  https://github.com/google-ai-edge/mediapipe-samples
- Casiez, Roussel, Vogel (2012), *1€ Filter: A Simple Speed-based Low-pass
  Filter for Noisy Input in Interactive Systems*, CHI 2012.
