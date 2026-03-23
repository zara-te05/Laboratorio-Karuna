from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns


# ══════════════════════════════════════════════════════════════════
# PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════
def preparar_datos(df: pd.DataFrame):
    df       = df.copy()
    X        = df[["cal_final", "prom_tareas", "prom_asist"]].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled


# ══════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════
def clustering(X_scaled, eps: float = 0.8, min_samples: int = 3):
    labels     = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    sil_score  = silhouette_score(X_scaled, labels) \
                 if n_clusters > 1 and (labels != -1).sum() > 1 else None
    return labels, n_clusters, n_outliers, sil_score


# ══════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE PERFILES
# Cada cluster recibe una de 4 etiquetas según sus promedios:
#   Alto rendimiento   — cal >= 80 y tareas >= 75
#   Medio rendimiento  — cal >= 65
#   Bajo rendimiento   — cal <  65 o tareas < 50
#   Atípico            — outlier DBSCAN (label == -1)
# ══════════════════════════════════════════════════════════════════
PERFILES = {
    "alto":   {"label": "Alto rendimiento",  "color": "#4CAF50"},
    "medio":  {"label": "Medio rendimiento", "color": "#2196F3"},
    "bajo":   {"label": "Bajo rendimiento",  "color": "#F44336"},
    "atipico":{"label": "Atípico",           "color": "#9E9E9E"},
}

def clasificar_perfil(cal, tareas):
    if cal >= 80 and tareas >= 75:
        return "alto"
    elif cal >= 65:
        return "medio"
    else:
        return "bajo"

def etiquetar_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels
    mapa_perfil = {}   # cluster_id → clave de perfil

    for c in sorted(set(labels)):
        if c == -1:
            mapa_perfil[c] = "atipico"
            continue
        sub    = df[df["cluster"] == c]
        cal    = sub["cal_final"].mean()
        tareas = sub["prom_tareas"].mean()
        mapa_perfil[c] = clasificar_perfil(cal, tareas)

    df["perfil"]   = df["cluster"].map(mapa_perfil)
    df["color"]    = df["perfil"].map(lambda p: PERFILES[p]["color"])
    df["etiqueta"] = df["perfil"].map(lambda p: PERFILES[p]["label"])
    return df, mapa_perfil


# ══════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════
def graficar(df, X_scaled, labels, n_clusters, n_outliers,
             sil_score, mapa_perfil, nombre_grupo, eps, min_samples):

    n            = len(df)
    point_colors = df["color"].values

    pca     = PCA(n_components=2)
    X_pca   = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    # Leyenda fija con los 4 perfiles
    leyenda_patches = [
        mpatches.Patch(color=PERFILES[k]["color"], label=PERFILES[k]["label"])
        for k in ["alto", "medio", "bajo", "atipico"]
    ]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"{nombre_grupo}  ·  DBSCAN  ·  eps={eps}  min_samples={min_samples}"
        f"  |  {n_clusters} clusters  ·  {n_outliers} atípicos",
        fontsize=13, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # ── Fila 0: panorama del grupo ────────────────────────────────

    # Distribución calificación final
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["cal_final"], bins=12, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.axvline(df["cal_final"].mean(), color="tomato", linestyle="--",
               linewidth=1.5, label=f"Media ({df['cal_final'].mean():.1f})")
    ax.axvline(80, color="#4CAF50", linestyle=":", linewidth=1.2, label="Alto (80)")
    ax.axvline(65, color="#F44336", linestyle=":", linewidth=1.2, label="Bajo (65)")
    ax.set_title("Dist. calificación final")
    ax.set_xlabel("Calificación"); ax.set_ylabel("Alumnos")
    ax.legend(fontsize=7)

    # Conteo por perfil
    ax = fig.add_subplot(gs[0, 1])
    conteo_perfil = df["perfil"].value_counts().reindex(
        ["alto", "medio", "bajo", "atipico"], fill_value=0)
    bar_colors = [PERFILES[p]["color"] for p in conteo_perfil.index]
    bar_labels = [PERFILES[p]["label"] for p in conteo_perfil.index]
    ax.bar(range(4), conteo_perfil.values, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(4))
    ax.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=8)
    ax.set_title("Alumnos por perfil")
    ax.set_ylabel("Cantidad")
    for i, v in enumerate(conteo_perfil.values):
        if v > 0:
            ax.text(i, v + 0.2, str(v), ha="center",
                    fontsize=10, fontweight="bold")

    # Boxplot variables por perfil
    ax = fig.add_subplot(gs[0, 2])
    df_melt = df.melt(id_vars="etiqueta",
                      value_vars=["cal_final", "prom_tareas", "prom_asist"],
                      var_name="variable", value_name="valor")
    df_melt["variable"] = df_melt["variable"].map({
        "cal_final": "Cal.", "prom_tareas": "Tareas", "prom_asist": "Asist."})
    palette_box = {PERFILES[k]["label"]: PERFILES[k]["color"]
                   for k in PERFILES}
    orden = [PERFILES[k]["label"] for k in ["alto", "medio", "bajo", "atipico"]
             if PERFILES[k]["label"] in df_melt["etiqueta"].unique()]
    sns.boxplot(data=df_melt, x="variable", y="valor", hue="etiqueta",
                hue_order=orden, palette=palette_box,
                ax=ax, linewidth=0.8)
    ax.set_title("Variables por perfil")
    ax.set_xlabel(""); ax.legend(title="", fontsize=7)

    # Correlación
    ax = fig.add_subplot(gs[0, 3])
    corr = df[["cal_final", "prom_tareas", "prom_asist"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.4, cbar=False,
                xticklabels=["Cal.", "Tareas", "Asist."],
                yticklabels=["Cal.", "Tareas", "Asist."],
                annot_kws={"size": 9})
    ax.set_title("Correlación")

    # ── Fila 1: visualización espacial ───────────────────────────

    # PCA 2D
    ax = fig.add_subplot(gs[1, 0:2])
    ax.scatter(X_pca[:, 0], X_pca[:, 1],
               c=point_colors, edgecolors="k",
               linewidths=0.3, s=65, alpha=0.85)
    ax.legend(handles=leyenda_patches, fontsize=8, loc="best")
    ax.set_title(f"PCA 2D  ({var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% varianza)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Cal. final vs Asistencia
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(df["cal_final"], df["prom_asist"],
               c=point_colors, edgecolors="k",
               linewidths=0.3, s=55, alpha=0.85)
    ax.axvline(80, color="#4CAF50", linestyle=":", linewidth=1)
    ax.axvline(65, color="#F44336", linestyle=":", linewidth=1)
    ax.set_title("Cal. final vs Asistencia")
    ax.set_xlabel("Calificación final"); ax.set_ylabel("Asistencia")
    ax.legend(handles=leyenda_patches, fontsize=7)

    # Cal. final vs Tareas
    ax = fig.add_subplot(gs[1, 3])
    ax.scatter(df["cal_final"], df["prom_tareas"],
               c=point_colors, edgecolors="k",
               linewidths=0.3, s=55, alpha=0.85)
    ax.axvline(80, color="#4CAF50", linestyle=":", linewidth=1)
    ax.axvline(65, color="#F44336", linestyle=":", linewidth=1)
    ax.axhline(75, color="#4CAF50", linestyle=":", linewidth=1)
    ax.axhline(50, color="#F44336", linestyle=":", linewidth=1)
    ax.set_title("Cal. final vs Tareas")
    ax.set_xlabel("Calificación final"); ax.set_ylabel("Tareas")

    # ── Fila 2: perfiles detallados ───────────────────────────────

    # Perfil promedio por categoría (barras agrupadas)
    ax = fig.add_subplot(gs[2, 0])
    variables  = ["Cal. final", "Tareas", "Asist."]
    cols_raw   = ["cal_final", "prom_tareas", "prom_asist"]
    perfiles_presentes = [p for p in ["alto", "medio", "bajo", "atipico"]
                          if p in df["perfil"].values]
    x_pos  = np.arange(len(variables))
    ancho  = 0.8 / len(perfiles_presentes)
    for i, p in enumerate(perfiles_presentes):
        sub    = df[df["perfil"] == p]
        medias = [sub[c].mean() for c in cols_raw]
        offset = (i - len(perfiles_presentes) / 2) * ancho + ancho / 2
        ax.bar(x_pos + offset, medias, width=ancho,
               color=PERFILES[p]["color"], edgecolor="white",
               label=PERFILES[p]["label"])
    ax.set_xticks(x_pos); ax.set_xticklabels(variables)
    ax.set_title("Perfil promedio por categoría")
    ax.set_ylabel("Promedio"); ax.set_ylim(0, 115)
    ax.axhline(80, color="#4CAF50", linestyle=":", linewidth=0.8)
    ax.axhline(65, color="#F44336", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=7)

    # Silhouette score
    ax = fig.add_subplot(gs[2, 1])
    if sil_score is not None:
        color_sil = "#4CAF50" if sil_score >= 0.5 else \
                    "#FF9800" if sil_score >= 0.3 else "#F44336"
        ax.barh([""], [sil_score], color=color_sil, height=0.4)
        ax.barh([""], [1], color="#e0e0e0", height=0.4, zorder=0)
        ax.barh([""], [sil_score], color=color_sil, height=0.4, zorder=1)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="orange", linestyle="--",
                   linewidth=1.2, label="Aceptable (0.5)")
        ax.axvline(0.7, color="green", linestyle="--",
                   linewidth=1.2, label="Bueno (0.7)")
        ax.text(sil_score + 0.02, 0, f"{sil_score:.3f}",
                va="center", fontsize=13, fontweight="bold")
        ax.set_title("Silhouette Score")
        ax.legend(fontsize=8); ax.set_yticks([])
    else:
        ax.text(0.5, 0.5,
                "No disponible\n(1 cluster o todos atípicos)",
                ha="center", va="center", color="gray", fontsize=9)
        ax.set_title("Silhouette Score")
        ax.set_yticks([])

    # Calificación por alumno coloreada por perfil
    ax = fig.add_subplot(gs[2, 2:4])
    idx_sort = np.argsort(df["cal_final"].values)
    bar_cols = [point_colors[i] for i in idx_sort]
    ax.barh(range(n), df["cal_final"].values[idx_sort],
            color=bar_cols, edgecolor="white", height=0.7)
    ax.axvline(80, color="#4CAF50", linestyle="--",
               linewidth=1.2, label="Alto (≥80)")
    ax.axvline(65, color="#F44336", linestyle="--",
               linewidth=1.2, label="Bajo (<65)")
    ax.legend(handles=leyenda_patches + [
        mpatches.Patch(color="none", label=""),
        mpatches.Patch(color="#4CAF50", label="Umbral alto (80)"),
        mpatches.Patch(color="#F44336", label="Umbral bajo (65)"),
    ], fontsize=7, loc="lower right")
    ax.set_title("Calificación final por alumno — coloreado por perfil\n"
                 "(ordenado de menor a mayor)")
    ax.set_xlabel("Calificación final")
    ax.set_yticks([])

    slug = f"{nombre_grupo}_dbscan".replace(" ", "_").lower()
    plt.savefig(f"{slug}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {slug}.png\n")


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def analizar_grupo(df: pd.DataFrame,
                   nombre_grupo: str = "Grupo",
                   eps: float = 0.8,
                   min_samples: int = 3):
    """
    Parámetros
    ----------
    df           : DataFrame con columnas cal_final, prom_tareas, prom_asist
    nombre_grupo : etiqueta del grupo (viene del front)
    eps          : radio de vecindad DBSCAN
    min_samples  : mínimo de puntos para formar un cluster
    """
    df, X, X_scaled = preparar_datos(df)
    labels, n_clusters, n_outliers, sil_score = clustering(X_scaled, eps, min_samples)
    df, mapa_perfil = etiquetar_clusters(df, labels)

    print(f"\n{'═'*50}")
    print(f"  {nombre_grupo} — DBSCAN")
    print(f"{'═'*50}")
    print(f"  Clusters encontrados : {n_clusters}")
    print(f"  Alumnos atípicos     : {n_outliers}")
    print(f"  Silhouette Score     : "
          f"{sil_score:.3f}" if sil_score else "  Silhouette Score     : N/A")

    for perfil_key in ["alto", "medio", "bajo", "atipico"]:
        sub = df[df["perfil"] == perfil_key]
        if len(sub) == 0:
            continue
        print(f"\n  ── {PERFILES[perfil_key]['label']} ({len(sub)} alumnos)")
        print(f"     Cal. final : {sub['cal_final'].mean():.1f}")
        print(f"     Tareas     : {sub['prom_tareas'].mean():.1f}")
        print(f"     Asistencia : {sub['prom_asist'].mean():.1f}")
    print()

    graficar(df, X_scaled, labels, n_clusters, n_outliers,
             sil_score, mapa_perfil, nombre_grupo, eps, min_samples)

    return df, labels, mapa_perfil


# ══════════════════════════════════════════════════════════════════
# SIMULACIÓN
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)

def grupo_aleatorio(n):
    return pd.DataFrame({
        "cal_final":   np.random.randint(50, 101, n),
        "prom_tareas": np.random.randint(30, 101, n),
        "prom_asist":  np.random.randint(50, 101, n),
    })

analizar_grupo(grupo_aleatorio(8),   nombre_grupo="Grupo pequeño")
analizar_grupo(grupo_aleatorio(35),  nombre_grupo="Grupo mediano")
analizar_grupo(grupo_aleatorio(120), nombre_grupo="Grupo grande")