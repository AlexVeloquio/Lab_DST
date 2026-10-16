"""
Simulador de la ESP32 Maestra. Habla el mismo protocolo que el firmware real.

    python esp32_simulador.py                 # escucha en 0.0.0.0:4210
    python esp32_simulador.py --puerto 4211
    python esp32_simulador.py --perdida 0.1   # tira el 10% de las tramas

Para que sirve
--------------
Permite desarrollar y demostrar la GUI y el enlace SIN la ESP32 armada. En la
misma maquina:

    terminal 1:  python esp32_simulador.py
    terminal 2:  python gui.py --host 127.0.0.1

El objetivo no es imitar los servos, sino validar la capa de comunicacion:
sincronia, CRC, numeros de secuencia, ACK y PING/PONG. Cuando llegue la ESP32
real basta cambiar --host por su IP; la GUI no distingue una de otra.

La opcion --perdida simula un enlace WiFi malo. Sirve para comprobar que los
reintentos del lado de la PC funcionan y que el failsafe entra cuando debe.
"""

from __future__ import annotations

import argparse
import random
import socket
import struct
import time

from core import protocol as P


class Simulador:
    def __init__(self, host: str, puerto: int, perdida: float,
                 verbose: bool = True) -> None:
        self.addr = (host, puerto)
        self.perdida = perdida
        self.verbose = verbose
        self.seq_tx = 0
        self.matriz = bytes(16)
        self.ultimo_rx = 0.0
        self.contadores: dict[str, int] = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.settimeout(0.5)

    # -- salida --------------------------------------------------------------
    def log(self, *a) -> None:
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}]", *a)

    def contar(self, nombre: str) -> None:
        self.contadores[nombre] = self.contadores.get(nombre, 0) + 1

    def enviar(self, tipo: P.MsgType, payload: bytes, destino) -> None:
        self.seq_tx = (self.seq_tx + 1) & 0xFF
        if self.perdida and random.random() < self.perdida:
            self.log(f"  (simulando perdida de {tipo.name})")
            return
        self.sock.sendto(P.build(tipo, self.seq_tx, payload), destino)

    # -- manejo de tramas ----------------------------------------------------
    def manejar(self, data: bytes, origen) -> None:
        f = P.parse(data)
        if f is None:
            self.contar("CRC_INVALIDO")
            self.log("trama descartada: sincronia o CRC invalidos")
            return

        if f.version != P.VERSION:
            self.enviar(P.MsgType.NACK,
                        bytes([f.seq, P.ErrCode.VERSION_INCOMPATIBLE]), origen)
            return

        self.ultimo_rx = time.time()
        nombre = f.type_name
        self.contar(nombre)

        if f.type == P.MsgType.PING:
            # Eco EXACTO del payload: es lo que permite a la PC distinguir su
            # propia respuesta de una vieja o de otro dispositivo.
            self.enviar(P.MsgType.PONG, f.payload, origen)
            marca = struct.unpack("<I", f.payload)[0] if len(f.payload) == 4 else 0
            self.log(f"PING  seq={f.seq} marca={marca} -> PONG")

        elif f.type == P.MsgType.HELLO:
            self.enviar(P.MsgType.ACK, bytes([f.seq]), origen)
            self.log(f"HELLO seq={f.seq} payload={f.payload!r} -> ACK")

        elif f.type == P.MsgType.STOP:
            self.matriz = bytes(16)
            self.enviar(P.MsgType.ACK, bytes([f.seq]), origen)
            self.log(f"STOP  seq={f.seq} -> ACK, modulos a reposo")

        elif f.type == P.MsgType.GESTURE:
            if len(f.payload) < 3:
                self.enviar(P.MsgType.NACK,
                            bytes([f.seq, P.ErrCode.LONGITUD_INVALIDA]), origen)
                return
            code, conf, extra = f.payload[0], f.payload[1], f.payload[2]
            nombres = {v: k for k, v in P.GESTURE_CODES.items()}
            signo = extra - 256 if extra > 127 else extra
            self.enviar(P.MsgType.ACK, bytes([f.seq]), origen)
            self.log(f"GESTO {nombres.get(code, hex(code))} "
                     f"conf={conf/255:.2f} extra={signo:+d} -> ACK")

        elif f.type == P.MsgType.MATRIX16:
            if len(f.payload) != 16:
                self.enviar(P.MsgType.NACK,
                            bytes([f.seq, P.ErrCode.LONGITUD_INVALIDA]), origen)
                return
            self.matriz = f.payload          # sin ACK: es flujo de estado

        elif f.type == P.MsgType.HEARTBEAT:
            pass                             # solo refresca ultimo_rx

        else:
            self.enviar(P.MsgType.NACK,
                        bytes([f.seq, P.ErrCode.TIPO_DESCONOCIDO]), origen)
            self.log(f"tipo desconocido {nombre} -> NACK")

    # -- bucle ---------------------------------------------------------------
    def correr(self) -> None:
        print(f"Simulador de ESP32 Maestra escuchando en "
              f"{self.addr[0]}:{self.addr[1]}")
        if self.perdida:
            print(f"Simulando {self.perdida*100:.0f} % de perdida de tramas")
        print("Ctrl+C para detener\n")

        t_resumen = time.time()
        try:
            while True:
                try:
                    data, origen = self.sock.recvfrom(512)
                    self.manejar(data, origen)
                except socket.timeout:
                    pass

                # Failsafe: igual que el firmware real
                if self.ultimo_rx and time.time() - self.ultimo_rx > 1.5:
                    if self.matriz != bytes(16):
                        self.matriz = bytes(16)
                        self.log("failsafe: 1.5 s sin trafico, modulos a reposo")

                if time.time() - t_resumen > 10:
                    t_resumen = time.time()
                    if self.contadores:
                        resumen = "  ".join(f"{k}:{v}" for k, v in
                                            sorted(self.contadores.items()))
                        nivel = max(self.matriz) if self.matriz else 0
                        self.log(f"resumen -> {resumen} | nivel max: {nivel}")
        except KeyboardInterrupt:
            print("\nSimulador detenido.")
        finally:
            self.sock.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulador de la ESP32 Maestra")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--puerto", type=int, default=4210)
    ap.add_argument("--perdida", type=float, default=0.0,
                    help="probabilidad de descartar una respuesta, 0 a 1")
    ap.add_argument("--silencio", action="store_true")
    args = ap.parse_args()
    Simulador(args.host, args.puerto, args.perdida,
              verbose=not args.silencio).correr()


if __name__ == "__main__":
    main()
