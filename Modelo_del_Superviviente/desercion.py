from lifelines import CoxPHFitter, KaplanMeierFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ══════════════════════════════════════════════════════════════════
# SIMULACIÓN
# Riesgo de deserción: depende principalmente de
# asistencia baja, seguido de calificación y tareas.
# ══════════════════════════════════════════════════════════════════
def simular_datos(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cal_final   = rng.integers(50, 101, n).astype(float)
    prom_tareas = rng.integers(30, 101, n).astype(float)
    prom_asist  = rng.integers(40, 101, n).astype(float)

    riesgo = (
        (100 - prom_asist)  / 100 * 0.5 +
        (100 - cal_final)   / 100 * 0.3 +
        (100 - prom_tareas) / 100 * 0.2
    )
    tiempo = np.clip(
        np.round(rng.exponential(1 / (riesgo + 0.05)) * 4), 1, 16
    ).astype(int)
    evento = (tiempo < 16).astype(int)
    # Alumnos con asistencia >= 80 rara vez desertan
    evento[prom_asist >= 80] = rng.choice(
        [0, 1], size=(prom_asist >= 80).sum(), p=[0.88, 0.12])

    return pd.DataFrame({
        "cal_final":   cal_final,
        "prom_tareas": prom_tareas,
        "prom_asist":  prom_asist,
        "tiempo":      tiempo,
        "evento":      evento,
    })


# ══════════════════════════════════════════════════════════════════
# NORMALIZACIÓN
# ══════════════════════════════════════════════════════════════════
def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["cal_final", "prom_tareas", "prom_asist"]:
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0
    return df


# ══════════════════════════════════════════════════════════════════
# AJUSTE DEL MODELO
# ══════════════════════════════════════════════════════════════════
def ajustar_modelo(df_norm: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_norm[["cal_final", "prom_tareas", "prom_asist",
                     "tiempo", "evento"]],
            duration_col="tiempo",
            event_col="evento",
            formula="cal_final + prom_tareas + prom_asist")
    return cph


# ══════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════
def graficar(df_orig, df_norm, cph, nombre_grupo):
    n        = len(df_orig)
    c_ev     = "#FF9800"   # naranja — desertó
    c_noev   = "#2196F3"   # azul    — permaneció
    c_cur    = "#FB8C00"
    features = ["Cal. final", "Tareas", "Asistencia"]

    summary = cph.summary
    hr      = np.exp(summary["coef"])
    ci_low  = np.exp(summary["coef lower 95%"])
    ci_high = np.exp(summary["coef upper 95%"])
    coefs   = summary["coef"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{nombre_grupo}  ·  Cox PH  ·  Deserción",
                 fontsize=13, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # ── Fila 0: panorama ─────────────────────────────────────────

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df_orig.loc[df_orig["evento"] == 0, "prom_asist"],
            bins=12, alpha=0.7, color=c_noev, label="Permaneció")
    ax.hist(df_orig.loc[df_orig["evento"] == 1, "prom_asist"],
            bins=12, alpha=0.7, color=c_ev,   label="Desertó")
    ax.set_title("Asistencia por estado")
    ax.set_xlabel("Promedio asistencia"); ax.set_ylabel("Alumnos")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    conteo = df_orig["evento"].value_counts().sort_index()
    ax.pie(conteo, labels=["Permaneció", "Desertó"],
           autopct="%1.1f%%",
           colors=[c_noev, c_ev], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    ax.set_title("Proporción deserción")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df_orig.loc[df_orig["evento"] == 1, "tiempo"],
            bins=16, color=c_ev,   alpha=0.7, label="Desertó")
    ax.hist(df_orig.loc[df_orig["evento"] == 0, "tiempo"],
            bins=16, color=c_noev, alpha=0.7, label="Permaneció")
    ax.set_title("Semanas hasta desertar")
    ax.set_xlabel("Semana"); ax.set_ylabel("Alumnos")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    corr = df_orig[["cal_final", "prom_tareas", "prom_asist", "evento"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.4, cbar=False,
                xticklabels=["Cal.", "Tareas", "Asist.", "Evento"],
                yticklabels=["Cal.", "Tareas", "Asist.", "Evento"],
                annot_kws={"size": 9})
    ax.set_title("Correlación")

    # ── Fila 1: resultados del modelo ────────────────────────────

    ax = fig.add_subplot(gs[1, 0])
    bar_cols_hr = [c_noev if h < 1 else c_ev for h in hr]
    ax.barh(features, hr, color=bar_cols_hr, edgecolor="white")
    ax.errorbar(hr, features,
                xerr=[hr - ci_low, ci_high - hr],
                fmt="none", color="black", capsize=4, linewidth=1)
    ax.axvline(1, color="black", linestyle="--", linewidth=1,
               label="HR=1 (sin efecto)")
    ax.set_title("Hazard Ratios\n(>1 = mayor riesgo, <1 = protege)")
    ax.set_xlabel("Hazard Ratio")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1:3])
    kmf = KaplanMeierFitter()
    kmf.fit(df_orig["tiempo"], event_observed=df_orig["evento"],
            label="Grupo completo")
    kmf.plot_survival_function(ax=ax, color=c_cur, ci_show=True)

    mask_alta = df_orig["prom_asist"] >= 80
    if mask_alta.sum() > 1:
        KaplanMeierFitter().fit(
            df_orig.loc[mask_alta, "tiempo"],
            event_observed=df_orig.loc[mask_alta, "evento"],
            label="Asist. ≥ 80"
        ).plot_survival_function(ax=ax, color=c_noev, ci_show=False)

    mask_baja = df_orig["prom_asist"] < 60
    if mask_baja.sum() > 1:
        KaplanMeierFitter().fit(
            df_orig.loc[mask_baja, "tiempo"],
            event_observed=df_orig.loc[mask_baja, "evento"],
            label="Asist. < 60"
        ).plot_survival_function(ax=ax, color=c_ev, ci_show=False)

    ax.set_title("Kaplan-Meier — P(no desertar) por semana")
    ax.set_xlabel("Semana"); ax.set_ylabel("P(permanencia)")
    ax.set_ylim(0, 1.05)

    ax = fig.add_subplot(gs[1, 3])
    bar_cols_coef = [c_noev if c < 0 else c_ev for c in coefs]
    ax.barh(features, coefs, color=bar_cols_coef, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Coeficientes log(HR)\n(<0 = protege, >0 = riesgo)")
    ax.set_xlabel("log(HR)")
    rng_coef = max(abs(coefs)) * 0.03
    for i, v in enumerate(coefs):
        ax.text(v + rng_coef, i, f"{v:.3f}", va="center", fontsize=9)

    # ── Fila 2: predicción individual ────────────────────────────

    ax = fig.add_subplot(gs[2, 0:2])
    riesgo_pred = cph.predict_partial_hazard(df_norm)
    idx_sort    = np.argsort(riesgo_pred.values)
    bar_cols_r  = [c_ev if df_orig["evento"].values[i] == 1
                   else c_noev for i in idx_sort]
    ax.barh(range(n), riesgo_pred.values[idx_sort],
            color=bar_cols_r, edgecolor="white", height=0.7)
    ax.set_title("Riesgo de deserción por alumno\n"
                 "(naranja=desertó, azul=permaneció)")
    ax.set_xlabel("Hazard parcial")
    ax.set_yticks([])

    ax = fig.add_subplot(gs[2, 2:4])
    perfiles_tipo = pd.DataFrame({
        "cal_final":   [0.9, 0.5, 0.1],
        "prom_tareas": [0.8, 0.5, 0.2],
        "prom_asist":  [0.9, 0.5, 0.2],
        "tiempo":      [16,  16,  16 ],
        "evento":      [0,   0,   0  ],
    })
    for color, label, (_, row) in zip(
        [c_noev, "mediumseagreen", c_ev],
        ["Perfil alto", "Perfil medio", "Perfil bajo"],
        perfiles_tipo.iterrows()
    ):
        sf = cph.predict_survival_function(row.to_frame().T)
        ax.plot(sf.index, sf.values.flatten(),
                color=color, linewidth=2, label=label)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("P(no desertar) predicha por perfil típico")
    ax.set_xlabel("Semana"); ax.set_ylabel("P(permanencia)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    slug = f"{nombre_grupo}_cox_desercion".replace(" ", "_").lower()
    plt.savefig(f"{slug}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {slug}.png\n")


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def analizar_grupo(df: pd.DataFrame, nombre_grupo: str = "Grupo"):
    """
    Parámetros
    ----------
    df           : DataFrame con columnas:
                   cal_final, prom_tareas, prom_asist, tiempo, evento
    nombre_grupo : etiqueta del grupo (viene del front)
    """
    df_orig = df.copy()
    df_norm = normalizar(df)
    cph     = ajustar_modelo(df_norm)

    print(f"\n{'═'*50}")
    print(f"  {nombre_grupo} — Cox PH — Deserción")
    print(f"{'═'*50}")
    cph.print_summary()

    graficar(df_orig, df_norm, cph, nombre_grupo)
    return cph


# ══════════════════════════════════════════════════════════════════
# SIMULACIÓN
# ══════════════════════════════════════════════════════════════════
analizar_grupo(simular_datos(35),  nombre_grupo="Grupo mediano")
analizar_grupo(simular_datos(120), nombre_grupo="Grupo grande")