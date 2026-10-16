"""
Analisis de las sesiones de evaluacion. Produce las tablas del reporte.

    python metricas.py resultados/*.csv
    python metricas.py resultados/*.csv --salida resultados/metricas.md

Lee uno o varios CSV generados por evaluacion.py y calcula tasa de acierto por
postura y por usuario, tasa por condicion de captura, matriz de confusion,
tiempo de respuesta, FPS y falsos positivos en reposo.

La salida son tablas en Markdown listas para pegar en el documento del
entregable, no graficas: lo que se evalua es un registro de metricas.

No requiere camara ni mediapipe, solo numpy. Se puede correr en cualquier
maquina a partir de los CSV.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict

import numpy as np

POSTURAS = ["PUNO", "PAZ", "SALUDO", "PULSO", "GIRO"]


def leer(rutas: list[str]) -> list[dict]:
    filas = []
    for ruta in rutas:
        with open(ruta, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                filas.append(r)
    return filas


def _f(valor: str, defecto=np.nan) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return defecto


def tabla(encabezados: list[str], filas: list[list]) -> str:
    out = ["| " + " | ".join(str(h) for h in encabezados) + " |",
           "| " + " | ".join("---" for _ in encabezados) + " |"]
    for fila in filas:
        out.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(out)


def pct(aciertos: int, total: int) -> str:
    return f"{100 * aciertos / total:.0f} % ({aciertos}/{total})" if total else "-"


# ---------------------------------------------------------------------------
def acierto_por_usuario(intentos: list[dict]) -> str:
    usuarios = sorted({r["usuario"] for r in intentos})
    filas = []
    for p in POSTURAS:
        fila = [p]
        tot_a = tot_n = 0
        for u in usuarios:
            sub = [r for r in intentos if r["usuario"] == u and r["esperada"] == p]
            a = sum(1 for r in sub if r["detectada"] == p)
            fila.append(pct(a, len(sub)))
            tot_a += a
            tot_n += len(sub)
        fila.append(pct(tot_a, tot_n))
        filas.append(fila)

    fila = ["**Media**"]
    for u in usuarios:
        sub = [r for r in intentos if r["usuario"] == u]
        fila.append(pct(sum(1 for r in sub if r["detectada"] == r["esperada"]),
                        len(sub)))
    fila.append(pct(sum(1 for r in intentos if r["detectada"] == r["esperada"]),
                    len(intentos)))
    filas.append(fila)
    return tabla(["Postura"] + usuarios + ["Media"], filas)


def acierto_por_condicion(intentos: list[dict]) -> str:
    ilums = sorted({r["iluminacion"] for r in intentos})
    dists = sorted({r["distancia_m"] for r in intentos}, key=lambda d: _f(d, 0))
    filas = []
    for il in ilums:
        fila = [il]
        for d in dists:
            sub = [r for r in intentos
                   if r["iluminacion"] == il and r["distancia_m"] == d]
            fila.append(pct(sum(1 for r in sub
                                if r["detectada"] == r["esperada"]), len(sub)))
        filas.append(fila)
    return tabla(["Iluminacion"] + [f"{d} m" for d in dists], filas)


def matriz_confusion(intentos: list[dict]) -> str:
    cols = POSTURAS + ["NINGUNA"]
    filas = []
    for p in POSTURAS:
        sub = [r for r in intentos if r["esperada"] == p]
        fila = [p]
        for c in cols:
            n = sum(1 for r in sub if r["detectada"] == c)
            fila.append(str(n) if n else "")
        filas.append(fila)
    return tabla(["Ejecutada \\ Detectada"] + cols, filas)


def latencias(intentos: list[dict]) -> str:
    filas = []
    for p in POSTURAS:
        v = np.array([_f(r["latencia_ms"]) for r in intentos
                      if r["esperada"] == p and r["detectada"] == p])
        v = v[~np.isnan(v)]
        if v.size:
            filas.append([p, f"{v.mean():.0f}", f"{v.std():.0f}",
                          f"{v.max():.0f}", str(v.size)])
        else:
            filas.append([p, "-", "-", "-", "0"])
    return tabla(["Postura", "Media (ms)", "Desv. (ms)", "Maximo (ms)", "n"],
                 filas)


def rendimiento(filas_csv: list[dict]) -> str:
    medio = np.array([_f(r["fps_medio"]) for r in filas_csv])
    p5 = np.array([_f(r["fps_p5"]) for r in filas_csv])
    medio, p5 = medio[~np.isnan(medio)], p5[~np.isnan(p5)]
    if not medio.size:
        return "_Sin datos de FPS._"
    return tabla(["Metrica", "Valor"],
                 [["FPS promedio", f"{medio.mean():.1f}"],
                  ["FPS percentil 5", f"{np.percentile(p5, 5):.1f}"],
                  ["FPS minimo observado", f"{p5.min():.1f}"]])


def falsos_positivos(reposo: list[dict]) -> str:
    if not reposo:
        return "_Sin ventanas de reposo registradas._"
    por_usuario = defaultdict(list)
    for r in reposo:
        por_usuario[r["usuario"]].append(r)

    filas = []
    for u, rs in sorted(por_usuario.items()):
        dur_min = max(_f(r["latencia_ms"], 0) for r in rs) / 60_000
        dur_min = max(dur_min, 1e-6)
        filas.append([u, f"{dur_min:.1f}", str(len(rs)),
                      f"{len(rs)/dur_min:.2f}"])

    conteo = defaultdict(int)
    for r in reposo:
        conteo[r["detectada"]] += 1
    reparto = ", ".join(f"{k}: {v}" for k, v in sorted(conteo.items(),
                                                       key=lambda x: -x[1]))
    return (tabla(["Usuario", "Duracion (min)", "Detecciones", "Eventos/min"],
                  filas)
            + f"\n\nReparto por postura: {reparto}")


# ---------------------------------------------------------------------------
def informe(filas_csv: list[dict]) -> str:
    intentos = [r for r in filas_csv if r["modo"] == "intento"]
    reposo = [r for r in filas_csv if r["modo"] == "reposo"]

    partes = ["# Registro de metricas de desempeno\n"]

    if intentos:
        total = len(intentos)
        aciertos = sum(1 for r in intentos if r["detectada"] == r["esperada"])
        fn = sum(1 for r in intentos if r["detectada"] == "NINGUNA")
        conf = total - aciertos - fn
        partes += [
            "## Resumen\n",
            tabla(["Metrica", "Valor"],
                  [["Intentos totales", total],
                   ["Tasa de acierto", pct(aciertos, total)],
                   ["Falsos negativos", pct(fn, total)],
                   ["Confusiones", pct(conf, total)],
                   ["Usuarios", len({r['usuario'] for r in intentos})]]),
            "\n## Tasa de acierto por postura y usuario\n",
            acierto_por_usuario(intentos),
            "\n## Tasa de acierto por condicion de captura\n",
            acierto_por_condicion(intentos),
            "\n## Matriz de confusion\n",
            matriz_confusion(intentos),
            "\n## Tiempo de respuesta\n",
            latencias(intentos),
        ]

    partes += ["\n## Rendimiento\n", rendimiento(filas_csv),
               "\n## Falsos positivos en reposo\n", falsos_positivos(reposo)]
    return "\n".join(partes) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analiza sesiones de evaluacion")
    ap.add_argument("csv", nargs="+", help="archivos CSV (admite comodines)")
    ap.add_argument("--salida", default=None, help="guardar el informe en .md")
    args = ap.parse_args()

    rutas: list[str] = []
    for patron in args.csv:
        rutas.extend(sorted(glob.glob(patron)) or [patron])

    filas = leer(rutas)
    if not filas:
        print("No hay datos en los archivos indicados.")
        sys.exit(1)

    texto = informe(filas)
    print(texto)
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"Informe guardado en {args.salida}")


if __name__ == "__main__":
    main()
