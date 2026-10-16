"""
Escaneo y PING real por BLE contra la ESP32 Maestra.

    python ble_ping.py --escanear               # solo listar lo que hay cerca
    python ble_ping.py                          # buscar sOFIa-Maestra y hacer ping
    python ble_ping.py --nombre MiESP32
    python ble_ping.py --direccion AA:BB:CC:DD:EE:FF --repeticiones 20

DIFERENCIA CON UN ESCANEO SIMPLE
--------------------------------
Un escaneo con BleakScanner.discover() solo LEE los anuncios que los
dispositivos emiten al aire. No se conecta ni les manda nada: equivale a ver
el nombre de una red WiFi sin conectarse. Sirve para confirmar que la ESP32
esta encendida y al alcance, nada mas.

Para que haya comunicacion de verdad hacen falta tres pasos mas:

    1. Conectarse al dispositivo (enlace GATT).
    2. Suscribirse a la caracteristica de notificacion, por donde contesta.
    3. Escribir la trama en la caracteristica de recepcion.

Eso es lo que hace este script. El PING que envia es el mismo del estandar de
comunicacion del proyecto: la ESP32 devuelve el payload exacto en un PONG y
aqui se mide el tiempo de ida y vuelta.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from core.ble_link import BleLink, NOMBRE_POR_DEFECTO, NUS_SERVICE, escanear


def listar(timeout: float, solo: bool) -> None:
    print(f"Buscando dispositivos BLE durante {timeout:.0f} s...\n")
    try:
        dispositivos = escanear(timeout=timeout, solo_sofia=solo)
    except Exception as exc:
        print(f"Error al escanear: {exc}")
        print("En macOS hay que dar permiso de Bluetooth a la terminal en "
              "Ajustes > Privacidad y seguridad > Bluetooth.")
        return

    if not dispositivos:
        print("No se encontro ningun dispositivo.")
        return

    compatibles = [d for d in dispositivos if d["compatible"]]
    print(f"{len(dispositivos)} dispositivos, {len(compatibles)} compatibles "
          f"con el protocolo de sOFIa:\n")

    for d in dispositivos:
        marca = "[sOFIa]" if d["compatible"] else "       "
        print(f"{marca} {d['nombre']}")
        print(f"         direccion: {d['direccion']}")
        print(f"         RSSI: {d['rssi']} dBm")
        if d["compatible"]:
            print("         expone el servicio Nordic UART: puede hablar "
                  "nuestro protocolo")
        print()

    if not compatibles:
        print("Ninguno anuncia el servicio Nordic UART "
              f"({NUS_SERVICE}).")
        print("Si tu ESP32 aparece en la lista pero no esta marcada, le falta "
              "el firmware BLE: sube firmware/esp32_master_ble/.")


def hacer_ping(nombre: str | None, direccion: str | None,
               repeticiones: int) -> int:
    enlace = BleLink(nombre=nombre or NOMBRE_POR_DEFECTO, direccion=direccion)

    destino = direccion or nombre or NOMBRE_POR_DEFECTO
    print(f"Conectando con {destino} ...")
    t0 = time.perf_counter()
    if not enlace.conectar():
        print("\nNo se pudo conectar.")
        print("Revisa que la ESP32 este encendida, con el firmware BLE, y que "
              "no este emparejada con otro equipo.")
        print("Para ver que hay cerca:  python ble_ping.py --escanear")
        enlace.close()
        return 1

    print(f"Conectado en {(time.perf_counter()-t0):.1f} s  ·  "
          f"MTU negociado: {enlace.mtu} bytes")
    if enlace.mtu < 27:
        print("  Aviso: con MTU 23 solo caben 20 bytes por notificacion y las "
              "tramas MATRIX16 viajaran partidas. Funciona, pero conviene "
              "subir el MTU en el firmware.")

    print("\nHandshake HELLO ...", end=" ", flush=True)
    print("ACK recibido" if enlace.hello() else "sin respuesta")

    print(f"\nEnviando {repeticiones} PING:\n")
    muestras: list[float] = []
    for i in range(repeticiones):
        rtt = enlace.ping()
        if rtt is None:
            print(f"  {i+1:3d}  sin respuesta")
        else:
            muestras.append(rtt)
            print(f"  {i+1:3d}  PONG en {rtt:7.1f} ms")
        time.sleep(0.2)

    perdidos = repeticiones - len(muestras)
    print(f"\n--- resumen ---")
    print(f"enviados {repeticiones}  ·  recibidos {len(muestras)}  ·  "
          f"perdidos {perdidos} ({100*perdidos/repeticiones:.0f} %)")
    if muestras:
        print(f"RTT  min {min(muestras):.1f}  ·  medio "
              f"{statistics.mean(muestras):.1f}  ·  max {max(muestras):.1f} ms")
        if len(muestras) > 1:
            print(f"desviacion {statistics.stdev(muestras):.1f} ms")

    enlace.close()
    return 0 if muestras else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Escaneo y ping BLE contra la ESP32 Maestra")
    ap.add_argument("--escanear", action="store_true",
                    help="solo listar dispositivos, sin conectarse")
    ap.add_argument("--nombre", default=None,
                    help=f"nombre anunciado (por defecto {NOMBRE_POR_DEFECTO})")
    ap.add_argument("--direccion", default=None,
                    help="direccion o UUID, salta el escaneo")
    ap.add_argument("--repeticiones", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--todos", action="store_true",
                    help="al escanear, mostrar tambien los no compatibles")
    args = ap.parse_args()

    if args.escanear:
        listar(args.timeout, solo=not args.todos)
        sys.exit(0)
    sys.exit(hacer_ping(args.nombre, args.direccion, args.repeticiones))


if __name__ == "__main__":
    main()
