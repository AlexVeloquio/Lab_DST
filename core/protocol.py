"""
Estandar de comunicacion PC <-> ESP32 Maestra (entregable de la semana 9).

DECISION DE TRANSPORTE: UDP para el flujo continuo, con ACK selectivo para los
comandos criticos.

  - El flujo MATRIX16 es de estado, no de eventos: si se pierde una trama, la
    siguiente (33 ms despues) ya trae el estado correcto. Retransmitir seria
    peor: TCP reordenaria y entregaria tarde un dato ya obsoleto, provocando el
    efecto "el panel va retrasado y luego se acelera".
  - Los comandos de evento (GESTURE, STOP) si importan: se envian con numero de
    secuencia y se espera ACK, con hasta 3 reintentos.

FORMATO DE TRAMA (binario, little-endian, longitud fija de cabecera):

   0      1      2      3      4      5          5+N     5+N+1
 +------+------+------+------+------+---------+------+------+
 | 0xA5 | 0x5A | VER  | TYPE | SEQ  | LEN | PAYLOAD | CRC16 |
 +------+------+------+------+------+---------+------+------+
   sync   sync    1B     1B     1B    1B    N bytes    2B

  VER     version del protocolo (0x01)
  TYPE    codigo de mensaje (ver clase MsgType)
  SEQ     contador 0-255 con vuelta; permite descartar tramas viejas
  LEN     longitud del payload (0-64)
  CRC16   CRC-16/CCITT-FALSE sobre VER..PAYLOAD

El doble byte de sincronia permite resincronizar el parser de la ESP32 si se
pierde un byte, sin necesidad de reiniciar el enlace.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from enum import IntEnum

SYNC = b"\xA5\x5A"
VERSION = 0x01
MAX_PAYLOAD = 64
HEADER_LEN = 6          # SYNC(2) + VER + TYPE + SEQ + LEN
CRC_LEN = 2


class MsgType(IntEnum):
    HELLO = 0x01        # PC -> ESP32, descubrimiento / handshake
    ACK = 0x02          # ESP32 -> PC, payload = SEQ confirmado
    NACK = 0x03         # ESP32 -> PC, payload = SEQ + codigo de error
    PING = 0x04         # PC -> ESP32, prueba de enlace; se mide el RTT
    PONG = 0x05         # ESP32 -> PC, eco exacto del payload del PING
    GESTURE = 0x10      # PC -> ESP32, evento de postura (requiere ACK)
    MATRIX16 = 0x11     # PC -> ESP32, 16 bytes de nivel por modulo (sin ACK)
    HEARTBEAT = 0x20    # PC -> ESP32 cada 500 ms; su ausencia dispara failsafe
    STATUS = 0x21       # ESP32 -> PC, telemetria
    STOP = 0x40         # PC -> ESP32, paro seguro (requiere ACK)


class ErrCode(IntEnum):
    OK = 0x00
    CRC_INVALIDO = 0x01
    TIPO_DESCONOCIDO = 0x02
    LONGITUD_INVALIDA = 0x03
    BUS_OCUPADO = 0x04
    ESCLAVO_SIN_RESPUESTA = 0x05
    VERSION_INCOMPATIBLE = 0x06
    PAYLOAD_INVALIDO = 0x07


# Codigos de gesto que viajan en el payload de GESTURE.
# Payload = [codigo, confianza 0-255, extra]
#
# El byte `extra` se interpreta como ENTERO CON SIGNO (int8) en la ESP32:
#   GIRO   -1 quedo mostrando el DORSO, +1 quedo mostrando la PALMA
#   los demas gestos lo dejan en 0
GESTURE_CODES = {
    "PUNO": 0x01,       # estatica
    "PAZ": 0x02,        # estatica
    "SALUDO": 0x03,     # dinamica
    "PULSO": 0x04,      # dinamica
    "GIRO": 0x05,       # dinamica
}


def crc16_ccitt(data: bytes, seed: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE. Implementacion bit a bit: 20 bytes por trama, sobra."""
    crc = seed
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build(msg_type: MsgType, seq: int, payload: bytes = b"") -> bytes:
    """Arma una trama completa lista para enviar."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload de {len(payload)} B excede {MAX_PAYLOAD}")
    body = struct.pack("<BBBB", VERSION, int(msg_type), seq & 0xFF, len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16_ccitt(body))


@dataclass
class Frame:
    version: int
    type: int
    seq: int
    payload: bytes

    @property
    def type_name(self) -> str:
        try:
            return MsgType(self.type).name
        except ValueError:
            return f"0x{self.type:02X}"


def parse(data: bytes) -> Frame | None:
    """
    Valida y decodifica una trama. Devuelve None si es invalida.
    Busca el patron de sincronia en vez de asumir que empieza en 0: asi el
    mismo parser sirve para UDP (una trama por datagrama) y para un flujo
    serie continuo.
    """
    i = data.find(SYNC)
    if i < 0 or len(data) - i < HEADER_LEN + CRC_LEN:
        return None

    ver, typ, seq, ln = struct.unpack_from("<BBBB", data, i + 2)
    end = i + HEADER_LEN + ln
    if ln > MAX_PAYLOAD or len(data) < end + CRC_LEN:
        return None

    body = data[i + 2:end]
    (crc_rx,) = struct.unpack_from("<H", data, end)
    if crc16_ccitt(body) != crc_rx:
        return None

    return Frame(ver, typ, seq, data[i + HEADER_LEN:end])


class FrameStream:
    """
    Extrae tramas completas de un flujo de bytes que puede venir partido.

    UDP entrega un datagrama entero por trama, asi que ahi `parse()` basta.
    BLE no: una notificacion cabe en MTU menos 3 bytes, y con el MTU por
    defecto de 23 eso son 20 bytes, menos que una trama MATRIX16 de 24. Las
    tramas llegan partidas en dos notificaciones, o dos tramas cortas llegan
    pegadas en una sola.

    Esta clase acumula lo que llega y devuelve las tramas que ya esten
    completas. Aprovecha el doble byte de sincronia para descartar basura
    inicial y recuperarse si se pierde un fragmento.
    """

    MAX_BUFFER = 4096

    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self.buf.extend(data)
        tramas: list[Frame] = []

        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                # Conservar el ultimo byte: puede ser la mitad de un SYNC
                self.buf = self.buf[-1:] if self.buf else bytearray()
                break
            if i:
                del self.buf[:i]                    # descartar basura previa
            if len(self.buf) < HEADER_LEN + CRC_LEN:
                break

            ln = self.buf[5]
            if ln > MAX_PAYLOAD:
                del self.buf[:2]                    # sincronia falsa
                continue
            total = HEADER_LEN + ln + CRC_LEN
            if len(self.buf) < total:
                break                               # falta cola, esperar

            f = parse(bytes(self.buf[:total]))
            if f is None:
                del self.buf[:2]                    # CRC malo: resincronizar
                continue
            tramas.append(f)
            del self.buf[:total]

        if len(self.buf) > self.MAX_BUFFER:
            self.buf = self.buf[-self.MAX_BUFFER:]
        return tramas

    def reset(self) -> None:
        self.buf.clear()


class SofiaLink:
    """
    Cliente UDP hacia la ESP32 Maestra.

    Maneja secuencia, reintentos con ACK, heartbeat y reconexion automatica
    (requisito de la semana 12). Es no bloqueante salvo en `send_reliable`.
    """

    def __init__(self, host: str, port: int = 4210, timeout: float = 0.25,
                 heartbeat_ms: int = 500) -> None:
        self.addr = (host, port)
        self.timeout = timeout
        self.heartbeat_ms = heartbeat_ms
        self.seq = 0
        self.connected = False
        self.last_rx_ms = 0
        self._last_hb = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)

    def _next_seq(self) -> int:
        self.seq = (self.seq + 1) & 0xFF
        return self.seq

    def close(self) -> None:
        self._sock.close()

    # -- envio ---------------------------------------------------------------
    def send_fast(self, msg_type: MsgType, payload: bytes = b"") -> None:
        """Dispara y olvida. Para MATRIX16 y HEARTBEAT."""
        try:
            self._sock.sendto(build(msg_type, self._next_seq(), payload), self.addr)
        except OSError:
            self.connected = False

    def send_reliable(self, msg_type: MsgType, payload: bytes = b"",
                      retries: int = 3) -> bool:
        """Envia y espera ACK con el mismo SEQ. Para GESTURE y STOP."""
        seq = self._next_seq()
        frame = build(msg_type, seq, payload)
        for _ in range(retries):
            try:
                self._sock.sendto(frame, self.addr)
                data, _ = self._sock.recvfrom(256)
            except (socket.timeout, OSError):
                continue
            f = parse(data)
            if f and f.type == MsgType.ACK and f.payload and f.payload[0] == seq:
                self.connected = True
                self.last_rx_ms = int(time.time() * 1000)
                return True
        self.connected = False
        return False

    # -- helpers de alto nivel ----------------------------------------------
    def hello(self) -> bool:
        self.connected = self.send_reliable(MsgType.HELLO, b"SOFIA-SC")
        return self.connected

    def ping(self, timeout: float | None = None) -> float | None:
        """
        Prueba de enlace. Devuelve el RTT en milisegundos, o None si no hubo
        respuesta.

        El payload lleva 4 bytes con una marca de tiempo en microsegundos, y
        la ESP32 debe
        devolverlos EXACTOS en un PONG. Comparar el eco, y no solo recibir
        algo, descarta dos fallas que de otro modo pasan por buenas: una
        respuesta vieja que llega tarde (de un ping anterior) y un dispositivo
        distinto contestando en el mismo puerto.

        El RTT se mide en la PC con un reloj monotono, asi que no depende de
        que los relojes de ambos extremos esten sincronizados.
        """
        t_out = self.timeout if timeout is None else timeout
        seq = self._next_seq()
        marca = int(time.perf_counter() * 1e6) & 0xFFFFFFFF
        payload = struct.pack("<I", marca)
        frame = build(MsgType.PING, seq, payload)

        anterior = self._sock.gettimeout()
        self._sock.settimeout(t_out)
        t0 = time.perf_counter()
        try:
            self._sock.sendto(frame, self.addr)
            # Puede llegar trafico de otro tipo primero (STATUS, ACK viejo):
            # se descarta hasta encontrar el PONG con nuestro eco, o agotar.
            while (time.perf_counter() - t0) < t_out:
                data, _ = self._sock.recvfrom(256)
                f = parse(data)
                if f and f.type == MsgType.PONG and f.payload == payload:
                    rtt = (time.perf_counter() - t0) * 1000.0
                    self.connected = True
                    self.last_rx_ms = int(time.time() * 1000)
                    return rtt
        except (socket.timeout, OSError):
            pass
        finally:
            self._sock.settimeout(anterior)

        self.connected = False
        return None

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
        """Llamar cada frame: mantiene el heartbeat y reintenta el handshake."""
        now = time.time()
        if (now - self._last_hb) * 1000 >= self.heartbeat_ms:
            self._last_hb = now
            if self.connected:
                self.send_fast(MsgType.HEARTBEAT)
            else:
                self.hello()
