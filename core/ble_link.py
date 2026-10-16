"""
Transporte BLE hacia la ESP32 Maestra. Misma interfaz que SofiaLink (UDP).

    from core.ble_link import BleLink
    enlace = BleLink(nombre="sOFIa-Maestra")
    enlace.hello()
    print(enlace.ping(), "ms")

POR QUE SE PUEDE CAMBIAR DE TRANSPORTE SIN TOCAR NADA MAS
---------------------------------------------------------
El estandar de comunicacion define una TRAMA, no un medio: sincronia, version,
tipo, secuencia, longitud, carga y CRC. Esos bytes son los mismos viajen por
UDP, por BLE o por un cable serie. Esta clase reimplementa solo el transporte
y expone los mismos metodos que SofiaLink, asi que la GUI, el motor de gestos
y el firmware comparten protocolo.

DIFERENCIAS REALES DE BLE FRENTE A UDP
--------------------------------------
1. ORIENTADO A CONEXION. Hay que descubrir el dispositivo, conectarse y
   suscribirse a notificaciones. No existe "mandar a una IP y ya".

2. TAMANO LIMITADO POR EL MTU. Por defecto el ATT MTU es 23 bytes, de los que
   quedan 20 utiles: menos que una trama MATRIX16 de 24. Se negocia un MTU
   mayor al conectar, y aun asi se reensambla con FrameStream por si el
   sistema operativo no lo concede.

3. ASINCRONO. Bleak es asyncio y el resto del proyecto es sincrono. Aqui se
   levanta un bucle de asyncio en un hilo propio y los metodos publicos le
   pasan corrutinas, de modo que quien llama no se entera.

4. LATENCIA DISTINTA. El intervalo de conexion BLE suele ser de 15 a 45 ms, asi
   que el RTT tipico es mayor y mas regular que en WiFi. A cambio no depende
   de un router ni de configurar IPs, lo que para una instalacion artistica
   montada en cualquier sala es una ventaja grande.

UUID DEL SERVICIO
-----------------
Se usa el perfil Nordic UART (NUS), que es el de facto para flujos de bytes
sobre BLE y esta soportado por todas las bibliotecas de ESP32:

    Servicio  6E400001-B5A3-F393-E0A9-E50E24DCCA9E
    RX        6E400002-...  la PC ESCRIBE aqui   (la ESP32 recibe)
    TX        6E400003-...  la ESP32 NOTIFICA    (la PC recibe)

Los nombres RX y TX estan desde el punto de vista de la ESP32, que es la
convencion del perfil. Confundirlos es el error mas comun al cablear esto.
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time

from .protocol import (
    MsgType, Frame, FrameStream, GESTURE_CODES, build,
)

try:
    from bleak import BleakClient, BleakScanner
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Falta bleak. Instala con:  pip install bleak\n"
        "En Linux ademas hace falta BlueZ."
    ) from exc

NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"      # PC escribe
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"      # PC recibe

NOMBRE_POR_DEFECTO = "sOFIa-Maestra"


# ---------------------------------------------------------------------------
def escanear(timeout: float = 8.0, solo_sofia: bool = False) -> list[dict]:
    """
    Lista los dispositivos BLE cercanos. Version sincrona, para la GUI y la
    linea de comandos.

    A diferencia de un escaneo simple, marca cuales anuncian el servicio NUS:
    esos son los que pueden hablar nuestro protocolo. Un dispositivo que solo
    aparece con su nombre no sirve de nada hasta que exponga el servicio.
    """
    async def _run():
        hallados = await BleakScanner.discover(timeout=timeout, return_adv=True)
        out = []
        for _, (dev, adv) in hallados.items():
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            es_nus = NUS_SERVICE in uuids
            nombre = dev.name or adv.local_name or "(sin nombre)"
            if solo_sofia and not es_nus:
                continue
            out.append({
                "nombre": nombre,
                "direccion": dev.address,
                "rssi": adv.rssi,
                "servicios": adv.service_uuids or [],
                "compatible": es_nus,
            })
        out.sort(key=lambda d: (not d["compatible"], -(d["rssi"] or -999)))
        return out

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
class BleLink:
    """
    Cliente BLE con la misma API publica que SofiaLink:
    hello, ping, send_gesture, send_matrix, stop, tick, close, connected.
    """

    def __init__(self, nombre: str = NOMBRE_POR_DEFECTO,
                 direccion: str | None = None, timeout: float = 1.0,
                 heartbeat_ms: int = 500) -> None:
        self.nombre = nombre
        self.direccion = direccion
        self.timeout = timeout
        self.heartbeat_ms = heartbeat_ms

        self.seq = 0
        self.connected = False
        self.last_rx_ms = 0
        self.mtu = 23

        self._cliente: BleakClient | None = None
        self._stream = FrameStream()
        self._recibidas: "asyncio.Queue[Frame]" = None  # creada en el bucle
        self._last_hb = 0.0

        # Un bucle de asyncio propio, en su hilo: Bleak es asincrono y el
        # resto del proyecto no. Asi el cambio de transporte no obliga a
        # reescribir la GUI ni el motor de gestos.
        self._loop = asyncio.new_event_loop()
        self._hilo = threading.Thread(target=self._correr_loop, daemon=True)
        self._hilo.start()

    # -- infraestructura del bucle ---------------------------------------
    def _correr_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._recibidas = asyncio.Queue()
        self._loop.run_forever()

    def _ejecutar(self, corrutina, timeout: float | None = None):
        """Lanza una corrutina en el bucle de BLE y espera su resultado."""
        fut = asyncio.run_coroutine_threadsafe(corrutina, self._loop)
        try:
            return fut.result(timeout or self.timeout + 10.0)
        except Exception:
            return None

    def _next_seq(self) -> int:
        self.seq = (self.seq + 1) & 0xFF
        return self.seq

    # -- conexion ---------------------------------------------------------
    def _al_recibir(self, _caracteristica, data: bytearray) -> None:
        """Callback de notificacion. Reensambla y encola las tramas completas."""
        for f in self._stream.feed(bytes(data)):
            self.last_rx_ms = int(time.time() * 1000)
            self._recibidas.put_nowait(f)

    async def _conectar(self) -> bool:
        objetivo = self.direccion
        if objetivo is None:
            dispositivo = await BleakScanner.find_device_by_name(
                self.nombre, timeout=8.0)
            if dispositivo is None:
                # Segundo intento: por servicio, por si el nombre no coincide
                hallados = await BleakScanner.discover(timeout=6.0, return_adv=True)
                for _, (dev, adv) in hallados.items():
                    if NUS_SERVICE in [u.lower() for u in (adv.service_uuids or [])]:
                        dispositivo = dev
                        break
            if dispositivo is None:
                return False
            objetivo = dispositivo.address
            self.direccion = objetivo

        cliente = BleakClient(objetivo)
        await cliente.connect()
        if not cliente.is_connected:
            return False

        await cliente.start_notify(NUS_TX, self._al_recibir)
        self._cliente = cliente
        self.mtu = getattr(cliente, "mtu_size", 23) or 23
        return True

    def conectar(self) -> bool:
        ok = bool(self._ejecutar(self._conectar(), timeout=25.0))
        self.connected = ok
        return ok

    async def _desconectar(self) -> None:
        if self._cliente is not None:
            try:
                await self._cliente.stop_notify(NUS_TX)
            except Exception:
                pass
            try:
                await self._cliente.disconnect()
            except Exception:
                pass
            self._cliente = None

    # -- envio ------------------------------------------------------------
    async def _escribir(self, datos: bytes, con_respuesta: bool) -> bool:
        if self._cliente is None or not self._cliente.is_connected:
            return False
        # Trocear si la trama excede lo que cabe en una escritura
        util = max(self.mtu - 3, 20)
        for i in range(0, len(datos), util):
            await self._cliente.write_gatt_char(
                NUS_RX, datos[i:i + util], response=con_respuesta)
        return True

    def send_fast(self, msg_type: MsgType, payload: bytes = b"") -> None:
        """Dispara y olvida. Para MATRIX16 y HEARTBEAT."""
        trama = build(msg_type, self._next_seq(), payload)
        asyncio.run_coroutine_threadsafe(
            self._escribir(trama, False), self._loop)

    async def _esperar(self, tipo: MsgType, comprobar, t_out: float):
        """Descarta tramas hasta encontrar la esperada, o agotar el tiempo."""
        limite = time.perf_counter() + t_out
        while True:
            restante = limite - time.perf_counter()
            if restante <= 0:
                return None
            try:
                f = await asyncio.wait_for(self._recibidas.get(), restante)
            except asyncio.TimeoutError:
                return None
            if f.type == tipo and comprobar(f):
                return f

    async def _fiable(self, msg_type: MsgType, payload: bytes,
                      reintentos: int) -> bool:
        for _ in range(reintentos):
            seq = self._next_seq()
            if not await self._escribir(build(msg_type, seq, payload), True):
                return False
            f = await self._esperar(MsgType.ACK,
                                    lambda fr: fr.payload and fr.payload[0] == seq,
                                    self.timeout)
            if f is not None:
                return True
        return False

    def send_reliable(self, msg_type: MsgType, payload: bytes = b"",
                      retries: int = 3) -> bool:
        ok = bool(self._ejecutar(self._fiable(msg_type, payload, retries),
                                 timeout=self.timeout * retries + 5.0))
        self.connected = ok
        return ok

    # -- API de alto nivel, identica a SofiaLink --------------------------
    def hello(self) -> bool:
        if self._cliente is None and not self.conectar():
            self.connected = False
            return False
        self.connected = self.send_reliable(MsgType.HELLO, b"SOFIA-SC")
        return self.connected

    async def _ping(self, t_out: float) -> float | None:
        marca = int(time.perf_counter() * 1e6) & 0xFFFFFFFF
        payload = struct.pack("<I", marca)
        t0 = time.perf_counter()
        if not await self._escribir(build(MsgType.PING, self._next_seq(), payload),
                                    False):
            return None
        f = await self._esperar(MsgType.PONG,
                                lambda fr: fr.payload == payload, t_out)
        return None if f is None else (time.perf_counter() - t0) * 1000.0

    def ping(self, timeout: float | None = None) -> float | None:
        """RTT en milisegundos, o None. Misma semantica de eco que en UDP."""
        t_out = self.timeout if timeout is None else timeout
        rtt = self._ejecutar(self._ping(t_out), timeout=t_out + 5.0)
        self.connected = rtt is not None
        return rtt

    def send_matrix(self, matrix_bytes: bytes) -> None:
        if len(matrix_bytes) != 16:
            raise ValueError("MATRIX16 requiere exactamente 16 bytes")
        self.send_fast(MsgType.MATRIX16, matrix_bytes)

    def send_gesture(self, name: str, confidence: float, extra: int = 0) -> bool:
        code = GESTURE_CODES.get(name)
        if code is None:
            return False
        conf = int(max(0, min(255, round(confidence * 255))))
        return self.send_reliable(MsgType.GESTURE, bytes([code, conf, extra & 0xFF]))

    def stop(self) -> bool:
        return self.send_reliable(MsgType.STOP)

    def tick(self) -> None:
        """Llamar cada cuadro: heartbeat y reconexion automatica."""
        ahora = time.time()
        if (ahora - self._last_hb) * 1000 < self.heartbeat_ms:
            return
        self._last_hb = ahora
        vivo = self._cliente is not None and self._cliente.is_connected
        if vivo:
            self.send_fast(MsgType.HEARTBEAT)
        else:
            self.connected = False
            self._cliente = None
            self._stream.reset()

    def close(self) -> None:
        self._ejecutar(self._desconectar(), timeout=6.0)
        self.connected = False
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._hilo.join(timeout=3.0)
