from sklearn.cluster import KMeans
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
# K-Means requiere definir k (número de clusters) de antemano.
# A diferencia de DBSCAN, no detecta outliers automáticamente —
# todos los puntos quedan asignados a algún cluster.
# ══════════════════════════════════════════════════════════════════
def clustering(X_scaled, k: int = 3):
    model  = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels) if k > 1 else None
    return labels, model.cluster_centers_, sil_score


# ══════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE PERFILES
# ══════════════════════════════════════════════════════════════════
PERFILES = {
    "alto":   {"label": "Alto rendimiento",  "color": "#4CAF50"},
    "medio":  {"label": "Medio rendimiento", "color": "#2196F3"},
    "bajo":   {"label": "Bajo rendimiento",  "color": "#F44336"},
}

def clasificar_perfil(cal):
    if cal >= 75:
        return "alto"
    elif cal >= 65:
        return "medio"
    else:
        return "bajo"

def etiquetar_clusters(df, labels):
    df = df.copy()
    df["cluster"]  = labels
    mapa_perfil = {}

    for c in sorted(set(labels)):
        sub = df[df["cluster"] == c]
        mapa_perfil[c] = clasificar_perfil(sub["cal_final"].mean())

    df["perfil"]   = df["cluster"].map(mapa_perfil)
    df["color"]    = df["perfil"].map(lambda p: PERFILES[p]["color"])
    df["etiqueta"] = df["perfil"].map(lambda p: PERFILES[p]["label"])
    return df, mapa_perfil


# ══════════════════════════════════════════════════════════════════
# ELBOW METHOD — ayuda a elegir el k óptimo
# ══════════════════════════════════════════════════════════════════
def elbow(X_scaled, k_max: int = 8):
    inercias = []
    sil_scores = []
    ks = range(2, min(k_max + 1, len(X_scaled)))
    for k in ks:
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        m.fit(X_scaled)
        inercias.append(m.inertia_)
        sil_scores.append(silhouette_score(X_scaled, m.labels_))
    return list(ks), inercias, sil_scores


# ══════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════
def graficar(df, X_scaled, labels, centers, sil_score,
             mapa_perfil, nombre_grupo, k):

    n            = len(df)
    point_colors = df["color"].values

    pca     = PCA(n_components=2)
    X_pca   = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(centers)
    var_exp = pca.explained_variance_ratio_

    ks, inercias, sil_scores = elbow(X_scaled)

    leyenda_patches = [
        mpatches.Patch(color=PERFILES[p]["color"], label=PERFILES[p]["label"])
        for p in ["alto", "medio", "bajo"]
    ]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"{nombre_grupo}  ·  K-Means  ·  k={k}",
        fontsize=13, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # ── Fila 0: panorama del grupo ────────────────────────────────

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["cal_final"], bins=12, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.axvline(df["cal_final"].mean(), color="tomato", linestyle="--",
               linewidth=1.5, label=f"Media ({df['cal_final'].mean():.1f})")
    ax.axvline(75, color="#4CAF50", linestyle=":", linewidth=1.2, label="Alto (75)")
    ax.axvline(65, color="#F44336", linestyle=":", linewidth=1.2, label="Bajo (65)")
    ax.set_title("Dist. calificación final")
    ax.set_xlabel("Calificación"); ax.set_ylabel("Alumnos")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    conteo_perfil = df["perfil"].value_counts().reindex(
        ["alto", "medio", "bajo"], fill_value=0)
    ax.bar(range(3), conteo_perfil.values,
           color=[PERFILES[p]["color"] for p in conteo_perfil.index],
           edgecolor="white")
    ax.set_xticks(range(3))
    ax.set_xticklabels([PERFILES[p]["label"] for p in conteo_perfil.index],
                       rotation=15, ha="right", fontsize=8)
    ax.set_title("Alumnos por perfil")
    ax.set_ylabel("Cantidad")
    for i, v in enumerate(conteo_perfil.values):
        if v > 0:
            ax.text(i, v + 0.2, str(v), ha="center",
                    fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    df_melt = df.melt(id_vars="etiqueta",
                      value_vars=["cal_final", "prom_tareas", "prom_asist"],
                      var_name="variable", value_name="valor")
    df_melt["variable"] = df_melt["variable"].map({
        "cal_final": "Cal.", "prom_tareas": "Tareas", "prom_asist": "Asist."})
    palette_box = {PERFILES[p]["label"]: PERFILES[p]["color"] for p in PERFILES}
    orden = [PERFILES[p]["label"] for p in ["alto", "medio", "bajo"]
             if PERFILES[p]["label"] in df_melt["etiqueta"].unique()]
    sns.boxplot(data=df_melt, x="variable", y="valor", hue="etiqueta",
                hue_order=orden, palette=palette_box, ax=ax, linewidth=0.8)
    ax.set_title("Variables por perfil")
    ax.set_xlabel(""); ax.legend(title="", fontsize=7)

    ax = fig.add_subplot(gs[0, 3])
    corr = df[["cal_final", "prom_tareas", "prom_asist"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.4, cbar=False,
                xticklabels=["Cal.", "Tareas", "Asist."],
                yticklabels=["Cal.", "Tareas", "Asist."],
                annot_kws={"size": 9})
    ax.set_title("Correlación")

    # ── Fila 1: visualización espacial ───────────────────────────

    # PCA 2D con centroides
    ax = fig.add_subplot(gs[1, 0:2])
    ax.scatter(X_pca[:, 0], X_pca[:, 1],
               c=point_colors, edgecolors="k",
               linewidths=0.3, s=60, alpha=0.85)
    # Centroides marcados con X
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
               c="black", marker="X", s=180, zorder=5, label="Centroide")
    ax.legend(handles=leyenda_patches + [
        mpatches.Patch(color="black", label="Centroide")], fontsize=8)
    ax.set_title(f"PCA 2D  ({var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% varianza)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(df["cal_final"], df["prom_asist"],
               c=point_colors, edgecolors="k",
               linewidths=0.3, s=55, alpha=0.85)
    ax.axvline(75, color="#4CAF50", linestyle=":", linewidth=1)
    ax.axvline(65, color="#F44336", linestyle=":", linewidth=1)
    ax.set_title("Cal. final vs Asistencia")
    ax.set_xlabel("Calificación final"); ax.set_ylabel("Asistencia")
    ax.legend(handles=leyenda_patches, fontsize=7)

    ax = fig.add_subplot(gs[1, 3])
    ax.scatter(df["cal_final"], df["prom_tareas"],
               c=point_colors, edgecolors="k",
               linewidths=0.3, s=55, alpha=0.85)
    ax.axvline(75, color="#4CAF50", linestyle=":", linewidth=1)
    ax.axvline(65, color="#F44336", linestyle=":", linewidth=1)
    ax.set_title("Cal. final vs Tareas")
    ax.set_xlabel("Calificación final"); ax.set_ylabel("Tareas")

    # ── Fila 2: elbow + perfiles + alumnos ───────────────────────

    # Elbow method (inercia + silhouette)
    ax  = fig.add_subplot(gs[2, 0])
    ax2 = ax.twinx()
    ax.plot(ks, inercias, color="steelblue", marker="o",
            markersize=5, linewidth=1.5, label="Inercia")
    ax2.plot(ks, sil_scores, color="darkorange", marker="s",
             markersize=5, linewidth=1.5, linestyle="--", label="Silhouette")
    ax.axvline(k, color="tomato", linestyle="--", linewidth=1.2,
               label=f"k actual ({k})")
    ax.set_title("Elbow method")
    ax.set_xlabel("k"); ax.set_ylabel("Inercia", color="steelblue")
    ax2.set_ylabel("Silhouette", color="darkorange")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # Silhouette score
    ax = fig.add_subplot(gs[2, 1])
    if sil_score is not None:
        color_sil = "#4CAF50" if sil_score >= 0.5 else \
                    "#FF9800" if sil_score >= 0.3 else "#F44336"
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
        ax.text(0.5, 0.5, "No disponible", ha="center",
                va="center", color="gray")
        ax.set_title("Silhouette Score")
        ax.set_yticks([])

    # Perfil promedio por categoría
    ax = fig.add_subplot(gs[2, 2])
    variables = ["Cal. final", "Tareas", "Asist."]
    cols_raw  = ["cal_final", "prom_tareas", "prom_asist"]
    perfiles_presentes = [p for p in ["alto", "medio", "bajo"]
                          if p in df["perfil"].values]
    x_pos = np.arange(len(variables))
    ancho = 0.8 / len(perfiles_presentes)
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
    ax.axhline(75, color="#4CAF50", linestyle=":", linewidth=0.8)
    ax.axhline(65, color="#F44336", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=7)

    # Calificación por alumno coloreada por perfil
    ax = fig.add_subplot(gs[2, 3])
    idx_sort = np.argsort(df["cal_final"].values)
    ax.barh(range(n), df["cal_final"].values[idx_sort],
            color=[point_colors[i] for i in idx_sort],
            edgecolor="white", height=0.7)
    ax.axvline(75, color="#4CAF50", linestyle="--", linewidth=1.2)
    ax.axvline(65, color="#F44336", linestyle="--", linewidth=1.2)
    ax.legend(handles=leyenda_patches, fontsize=7, loc="lower right")
    ax.set_title("Calificación por alumno — coloreado por perfil\n"
                 "(ordenado de menor a mayor)")
    ax.set_xlabel("Calificación final")
    ax.set_yticks([])

    slug = f"{nombre_grupo}_kmeans".replace(" ", "_").lower()
    plt.savefig(f"{slug}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {slug}.png\n")


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def analizar_grupo(df: pd.DataFrame,
                   nombre_grupo: str = "Grupo",
                   k: int = 3):
    """
    Parámetros
    ----------
    df           : DataFrame con columnas cal_final, prom_tareas, prom_asist
    nombre_grupo : etiqueta del grupo (viene del front)
    k            : número de clusters (3 = alto / medio / bajo)
    """
    df, X, X_scaled = preparar_datos(df)
    labels, centers, sil_score = clustering(X_scaled, k)
    df, mapa_perfil = etiquetar_clusters(df, labels)

    print(f"\n{'═'*50}")
    print(f"  {nombre_grupo} — K-Means  (k={k})")
    print(f"{'═'*50}")
    print(f"  Silhouette Score : "
          f"{sil_score:.3f}" if sil_score else "  Silhouette Score : N/A")

    for perfil_key in ["alto", "medio", "bajo"]:
        sub = df[df["perfil"] == perfil_key]
        if len(sub) == 0:
            continue
        print(f"\n  ── {PERFILES[perfil_key]['label']} ({len(sub)} alumnos)")
        print(f"     Cal. final : {sub['cal_final'].mean():.1f}")
        print(f"     Tareas     : {sub['prom_tareas'].mean():.1f}")
        print(f"     Asistencia : {sub['prom_asist'].mean():.1f}")
    print()

    graficar(df, X_scaled, labels, centers, sil_score,
             mapa_perfil, nombre_grupo, k)

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
