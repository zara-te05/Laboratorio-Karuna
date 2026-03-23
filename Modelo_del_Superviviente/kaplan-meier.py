from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════════
# SIMULACIÓN
# ══════════════════════════════════════════════════════════════════
def simular_datos(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cal_final   = rng.integers(50, 101, n).astype(float)
    prom_tareas = rng.integers(30, 101, n).astype(float)
    prom_asist  = rng.integers(40, 101, n).astype(float)

    # ── Reprobación ───────────────────────────────────────────────
    riesgo_rep = (100 - cal_final) / 100 * 0.6 + \
                 (100 - prom_tareas) / 100 * 0.4
    t_rep  = np.clip(np.round(
        rng.exponential(1 / (riesgo_rep + 0.05)) * 4), 1, 16).astype(int)
    ev_rep = (t_rep < 16).astype(int)
    ev_rep[cal_final >= 75] = rng.choice(
        [0, 1], size=(cal_final >= 75).sum(), p=[0.90, 0.10])

    # ── Deserción ─────────────────────────────────────────────────
    riesgo_des = (100 - prom_asist)  / 100 * 0.5 + \
                 (100 - cal_final)   / 100 * 0.3 + \
                 (100 - prom_tareas) / 100 * 0.2
    t_des  = np.clip(np.round(
        rng.exponential(1 / (riesgo_des + 0.05)) * 4), 1, 16).astype(int)
    ev_des = (t_des < 16).astype(int)
    ev_des[prom_asist >= 80] = rng.choice(
        [0, 1], size=(prom_asist >= 80).sum(), p=[0.88, 0.12])

    # ── Mejora ────────────────────────────────────────────────────
    riesgo_mej = prom_tareas / 100 * 0.5 + \
                 prom_asist  / 100 * 0.3 + \
                 cal_final   / 100 * 0.2
    t_mej  = np.clip(np.round(
        rng.exponential(1 / (riesgo_mej + 0.05)) * 3), 1, 16).astype(int)
    ev_mej = (t_mej < 16).astype(int)
    ev_mej[prom_tareas < 50] = rng.choice(
        [0, 1], size=(prom_tareas < 50).sum(), p=[0.85, 0.15])

    return pd.DataFrame({
        "cal_final":   cal_final,
        "prom_tareas": prom_tareas,
        "prom_asist":  prom_asist,
        "tiempo_rep":  t_rep,  "evento_rep": ev_rep,
        "tiempo_des":  t_des,  "evento_des": ev_des,
        "tiempo_mej":  t_mej,  "evento_mej": ev_mej,
    })


# ══════════════════════════════════════════════════════════════════
# SEGMENTACIÓN EN 3 GRUPOS POR VARIABLE
# bajo / medio / alto según terciles del grupo
# ══════════════════════════════════════════════════════════════════
def segmentar(serie: pd.Series) -> pd.Series:
    q33 = serie.quantile(0.33)
    q66 = serie.quantile(0.66)
    return pd.cut(serie, bins=[-np.inf, q33, q66, np.inf],
                  labels=["Bajo", "Medio", "Alto"])


# ══════════════════════════════════════════════════════════════════
# GRAFICAR UN PANEL KM  (una variable × un evento)
# ══════════════════════════════════════════════════════════════════
def panel_km(ax, df, tiempo_col, evento_col,
             segmento_col, colores, ylabel, titulo):
    """
    Dibuja 3 curvas KM (bajo/medio/alto) en un ax dado
    e imprime el p-valor del log-rank test.
    """
    grupos  = ["Bajo", "Medio", "Alto"]
    results = []

    for grupo, color in zip(grupos, colores):
        mask = df[segmento_col] == grupo
        if mask.sum() < 2:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, tiempo_col],
                event_observed=df.loc[mask, evento_col],
                label=grupo)
        kmf.plot_survival_function(ax=ax, color=color,
                                   ci_show=True, ci_alpha=0.08)
        results.append((df.loc[mask, tiempo_col],
                        df.loc[mask, evento_col]))

    # Log-rank test multivariado
    if len(results) >= 2:
        lr = multivariate_logrank_test(
            df[tiempo_col], df[segmento_col], df[evento_col])
        p  = lr.p_value
        sig = "***" if p < 0.001 else "**" if p < 0.01 else \
              "*"   if p < 0.05  else "ns"
        ax.set_title(f"{titulo}\np={p:.4f} {sig}", fontsize=9)
    else:
        ax.set_title(titulo, fontsize=9)

    ax.set_xlabel("Semana", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, title="Grupo", title_fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--",
               linewidth=0.7, alpha=0.6)


# ══════════════════════════════════════════════════════════════════
# GRÁFICA GLOBAL — curva KM por evento sin segmentar
# ══════════════════════════════════════════════════════════════════
def panel_km_global(ax, df, tiempo_col, evento_col, color, titulo, ylabel):
    kmf = KaplanMeierFitter()
    kmf.fit(df[tiempo_col], event_observed=df[evento_col],
            label="Grupo completo")
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel("Semana", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.axhline(0.5, color="gray", linestyle="--",
               linewidth=0.7, alpha=0.6)
    # Mediana de supervivencia
    mediana = kmf.median_survival_time_
    if not np.isinf(mediana):
        ax.axvline(mediana, color=color, linestyle=":",
                   linewidth=1.2, alpha=0.8,
                   label=f"Mediana: sem. {int(mediana)}")
        ax.legend(fontsize=7)


# ══════════════════════════════════════════════════════════════════
# FIGURA PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def graficar(df, nombre_grupo):

    # Segmentar las 3 variables
    df = df.copy()
    df["seg_cal"]    = segmentar(df["cal_final"])
    df["seg_tareas"] = segmentar(df["prom_tareas"])
    df["seg_asist"]  = segmentar(df["prom_asist"])

    # Paletas por evento
    pal_rep = ["#EF9A9A", "#E53935", "#B71C1C"]   # rojos
    pal_des = ["#FFCC80", "#FB8C00", "#E65100"]   # naranjas
    pal_mej = ["#A5D6A7", "#43A047", "#1B5E20"]   # verdes

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"{nombre_grupo}  ·  Kaplan-Meier  ·  "
                 f"Reprobación | Deserción | Mejora",
                 fontsize=13, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.65, wspace=0.38)

    # ── Fila 0: curvas globales por evento ────────────────────────
    panel_km_global(fig.add_subplot(gs[0, 0]), df,
                    "tiempo_rep", "evento_rep",
                    "#E53935", "Reprobación — grupo completo",
                    "P(no reprobar)")

    panel_km_global(fig.add_subplot(gs[0, 1]), df,
                    "tiempo_des", "evento_des",
                    "#FB8C00", "Deserción — grupo completo",
                    "P(no desertar)")

    panel_km_global(fig.add_subplot(gs[0, 2]), df,
                    "tiempo_mej", "evento_mej",
                    "#43A047", "Mejora — grupo completo",
                    "P(aún sin mejorar)")

    # ── Fila 1: segmentado por calificación ──────────────────────
    fig.text(0.01, 0.72, "Por\ncalificación",
             va="center", ha="left", fontsize=9,
             fontweight="bold", color="#1565C0", rotation=90)

    panel_km(fig.add_subplot(gs[1, 0]), df,
             "tiempo_rep", "evento_rep", "seg_cal",
             pal_rep, "P(no reprobar)", "Reprobación × Cal.")

    panel_km(fig.add_subplot(gs[1, 1]), df,
             "tiempo_des", "evento_des", "seg_cal",
             pal_des, "P(no desertar)", "Deserción × Cal.")

    panel_km(fig.add_subplot(gs[1, 2]), df,
             "tiempo_mej", "evento_mej", "seg_cal",
             pal_mej, "P(aún sin mejorar)", "Mejora × Cal.")

    # ── Fila 2: segmentado por asistencia ─────────────────────────
    fig.text(0.01, 0.50, "Por\nasistencia",
             va="center", ha="left", fontsize=9,
             fontweight="bold", color="#6A1B9A", rotation=90)

    panel_km(fig.add_subplot(gs[2, 0]), df,
             "tiempo_rep", "evento_rep", "seg_asist",
             pal_rep, "P(no reprobar)", "Reprobación × Asist.")

    panel_km(fig.add_subplot(gs[2, 1]), df,
             "tiempo_des", "evento_des", "seg_asist",
             pal_des, "P(no desertar)", "Deserción × Asist.")

    panel_km(fig.add_subplot(gs[2, 2]), df,
             "tiempo_mej", "evento_mej", "seg_asist",
             pal_mej, "P(aún sin mejorar)", "Mejora × Asist.")

    # ── Fila 3: segmentado por tareas ─────────────────────────────
    fig.text(0.01, 0.27, "Por\ntareas",
             va="center", ha="left", fontsize=9,
             fontweight="bold", color="#2E7D32", rotation=90)

    panel_km(fig.add_subplot(gs[3, 0]), df,
             "tiempo_rep", "evento_rep", "seg_tareas",
             pal_rep, "P(no reprobar)", "Reprobación × Tareas")

    panel_km(fig.add_subplot(gs[3, 1]), df,
             "tiempo_des", "evento_des", "seg_tareas",
             pal_des, "P(no desertar)", "Deserción × Tareas")

    panel_km(fig.add_subplot(gs[3, 2]), df,
             "tiempo_mej", "evento_mej", "seg_tareas",
             pal_mej, "P(aún sin mejorar)", "Mejora × Tareas")

    slug = f"{nombre_grupo}_kaplan_meier".replace(" ", "_").lower()
    plt.savefig(f"{slug}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {slug}.png\n")


# ══════════════════════════════════════════════════════════════════
# RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════
def resumen_consola(df, nombre_grupo):
    print(f"\n{'═'*55}")
    print(f"  {nombre_grupo} — Kaplan-Meier")
    print(f"{'═'*55}")

    eventos = [
        ("tiempo_rep", "evento_rep", "Reprobación"),
        ("tiempo_des", "evento_des", "Deserción"),
        ("tiempo_mej", "evento_mej", "Mejora"),
    ]
    for tiempo_col, evento_col, nombre_ev in eventos:
        kmf = KaplanMeierFitter()
        kmf.fit(df[tiempo_col], event_observed=df[evento_col])
        mediana = kmf.median_survival_time_
        tasa    = df[evento_col].mean()
        print(f"\n  ── {nombre_ev}")
        print(f"     Tasa de evento  : {tasa:.1%}")
        print(f"     Mediana semanas : "
              f"{int(mediana) if not np.isinf(mediana) else '>16'}")

    print()


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def analizar_grupo(df: pd.DataFrame, nombre_grupo: str = "Grupo"):
    """
    Parámetros
    ----------
    df           : DataFrame con columnas:
                   cal_final, prom_tareas, prom_asist,
                   tiempo_rep, evento_rep,
                   tiempo_des, evento_des,
                   tiempo_mej, evento_mej
    nombre_grupo : etiqueta del grupo (viene del front)
    """
    resumen_consola(df, nombre_grupo)
    graficar(df, nombre_grupo)


# ══════════════════════════════════════════════════════════════════
# SIMULACIÓN
# ══════════════════════════════════════════════════════════════════
analizar_grupo(simular_datos(35),  nombre_grupo="Grupo mediano")
analizar_grupo(simular_datos(120), nombre_grupo="Grupo grande")