from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def analizar_grupo(df: pd.DataFrame, nombre_grupo: str = "Grupo",
                   umbral_aprobatorio: int = 70):

    n = len(df)
    df = df.copy()
    df["aprueba"] = (df["cal_final"] >= umbral_aprobatorio).astype(int)

    X = df[["cal_final", "prom_tareas", "prom_asist"]]
    y = df["aprueba"]
    clases, conteos = np.unique(y, return_counts=True)

    modo       = ""
    scores_loo = None
    y_test_plot = y_pred_plot = y_prob_plot = None
    X_test_plot = None

    # ── Caso 1: datos escasos ─────────────────────────────────────
    if n < 15:
        modo = f"Datos escasos ({n} alumnos) — Leave-One-Out CV"
        loo   = LeaveOneOut()
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        scores_loo = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
        model.fit(X, y)
        y_pred_plot = model.predict(X)
        y_prob_plot = model.predict_proba(X)[:, 1]
        y_test_plot = y
        X_test_plot = X

    else:
        # ── Caso 2: desbalance ────────────────────────────────────
        min_ratio = conteos.min() / n
        if min_ratio < 0.20:
            modo = f"Desbalance detectado (ratio {min_ratio:.0%}) — Oversampling"
            df_majority = df[y == conteos.argmax()]
            df_minority = df[y == conteos.argmin()]
            df_minority_up = resample(df_minority, replace=True,
                                      n_samples=len(df_majority), random_state=42)
            df_bal = pd.concat([df_majority, df_minority_up])
            X = df_bal[["cal_final", "prom_tareas", "prom_asist"]]
            y = df_bal["aprueba"]
        else:
            modo = f"Normal ({n} alumnos) — Train/Test split"

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred_plot = model.predict(X_test)
        y_prob_plot = model.predict_proba(X_test)[:, 1]
        y_test_plot = y_test
        X_test_plot = X_test

    graficar_grupo(df, model, X_test_plot, y_test_plot,
                   y_pred_plot, y_prob_plot,
                   scores_loo, nombre_grupo, modo, umbral_aprobatorio)

    return model, df


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN DE GRAFICAS
# ══════════════════════════════════════════════════════════════════
def graficar_grupo(df, model, X_test, y_test, y_pred, y_prob,
                   scores_loo, nombre_grupo, modo, umbral):

    n     = len(df)
    color_ap  = "#4CAF50"
    color_rep = "#F44336"
    color_neu = "steelblue"

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{nombre_grupo}  ·  {n} alumnos  ·  {modo}",
                 fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Fila 0: panorama del grupo ────────────────────────────────

    # 0-A  Distribución calificación final
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(df["cal_final"], bins=12, color=color_neu,
             edgecolor="white", linewidth=0.5)
    ax0.axvline(umbral, color="orange", linestyle="--",
                linewidth=1.5, label=f"Mínimo ({umbral})")
    ax0.axvline(df["cal_final"].mean(), color="tomato", linestyle=":",
                linewidth=1.5, label=f"Media ({df['cal_final'].mean():.1f})")
    ax0.set_title("Dist. calificación final")
    ax0.set_xlabel("Calificación"); ax0.set_ylabel("Alumnos")
    ax0.legend(fontsize=8)

    # 0-B  Proporción aprueba / reprueba
    ax1 = fig.add_subplot(gs[0, 1])
    counts = df["aprueba"].value_counts().sort_index()
    labels = ["Reprueba", "Aprueba"]
    colors_pie = [color_rep, color_ap]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors_pie, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title("Aprueba / Reprueba")

    # 0-C  Boxplot por resultado
    ax2 = fig.add_subplot(gs[0, 2])
    df_melt = df.melt(id_vars="aprueba",
                      value_vars=["cal_final", "prom_tareas", "prom_asist"],
                      var_name="variable", value_name="valor")
    df_melt["variable"] = df_melt["variable"].map({
        "cal_final": "Cal.", "prom_tareas": "Tareas", "prom_asist": "Asist."})
    df_melt["resultado"] = df_melt["aprueba"].map({1: "Aprueba", 0: "Reprueba"})
    sns.boxplot(data=df_melt, x="variable", y="valor", hue="resultado",
                palette={"Aprueba": color_ap, "Reprueba": color_rep},
                ax=ax2, linewidth=0.8)
    ax2.set_title("Variables por resultado")
    ax2.set_xlabel(""); ax2.set_ylabel("Valor")
    ax2.legend(title="", fontsize=8)

    # 0-D  Correlación
    ax3 = fig.add_subplot(gs[0, 3])
    corr = df[["cal_final", "prom_tareas", "prom_asist", "aprueba"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax3, linewidths=0.4, cbar=False,
                xticklabels=["Cal.", "Tareas", "Asist.", "Aprueba"],
                yticklabels=["Cal.", "Tareas", "Asist.", "Aprueba"],
                annot_kws={"size": 8})
    ax3.set_title("Correlación")

    # ── Fila 1: modelo ────────────────────────────────────────────

    # 1-A  Matriz de confusión
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
                xticklabels=["Reprueba", "Aprueba"],
                yticklabels=["Reprueba", "Aprueba"],
                linewidths=0.5)
    ax4.set_title("Matriz de confusión")
    ax4.set_xlabel("Predicción"); ax4.set_ylabel("Real")

    # 1-B  Distribución de probabilidades
    ax5 = fig.add_subplot(gs[1, 1])
    y_test_arr = np.array(y_test)
    ax5.hist(y_prob[y_test_arr == 0], bins=10, alpha=0.7,
             color=color_rep, label="Reprueba real")
    ax5.hist(y_prob[y_test_arr == 1], bins=10, alpha=0.7,
             color=color_ap,  label="Aprueba real")
    ax5.axvline(0.5, color="black", linestyle="--", linewidth=1.2)
    ax5.set_title("Distribución P(aprueba)")
    ax5.set_xlabel("Probabilidad"); ax5.set_ylabel("Frecuencia")
    ax5.legend(fontsize=8)

    # 1-C  Curva ROC  (solo si hay ambas clases en y_test)
    ax6 = fig.add_subplot(gs[1, 2])
    clases_test = np.unique(y_test_arr)
    if len(clases_test) == 2:
        fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
        roc_auc = auc(fpr, tpr)
        ax6.plot(fpr, tpr, color=color_neu, lw=2,
                 label=f"AUC = {roc_auc:.3f}")
        ax6.fill_between(fpr, tpr, alpha=0.1, color=color_neu)
        ax6.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax6.legend(fontsize=9)
    else:
        ax6.text(0.5, 0.5, "Solo una clase\nen el set de prueba",
                 ha="center", va="center", fontsize=10, color="gray")
    ax6.set_title("Curva ROC")
    ax6.set_xlabel("FPR"); ax6.set_ylabel("TPR")

    # 1-D  Coeficientes
    ax7 = fig.add_subplot(gs[1, 3])
    features = ["Cal. final", "Tareas", "Asistencia"]
    coefs    = model.coef_[0]
    bar_colors = [color_ap if c > 0 else color_rep for c in coefs]
    ax7.barh(features, coefs, color=bar_colors, edgecolor="white")
    ax7.axvline(0, color="black", linewidth=0.8)
    ax7.set_title("Coeficientes del modelo")
    ax7.set_xlabel("Peso")
    for i, v in enumerate(coefs):
        ax7.text(v + (0.01 if v >= 0 else -0.01), i,
                 f"{v:.3f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9)

    # ── Fila 2: comportamiento individual ─────────────────────────

    # 2-A  Probabilidad predicha por alumno (todos)
    ax8 = fig.add_subplot(gs[2, 0:2])
    all_prob  = model.predict_proba(
        df[["cal_final", "prom_tareas", "prom_asist"]])[:, 1]
    idx_sort  = np.argsort(all_prob)
    bar_cols  = [color_ap if df["aprueba"].values[i] == 1
                 else color_rep for i in idx_sort]
    ax8.barh(range(n), all_prob[idx_sort], color=bar_cols,
             edgecolor="white", height=0.7)
    ax8.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax8.set_title("P(aprueba) por alumno — todo el grupo\n"
                  "(verde = aprueba real, rojo = reprueba real)")
    ax8.set_xlabel("Probabilidad predicha")
    ax8.set_yticks([])
    ax8.set_xlim(0, 1.05)

    # 2-B  Scatter cal_final vs asistencia
    ax9 = fig.add_subplot(gs[2, 2])
    scatter = ax9.scatter(
        df["cal_final"], df["prom_asist"],
        c=df["aprueba"], cmap="RdYlGn",
        edgecolors="k", linewidths=0.3, s=45, alpha=0.85)
    ax9.axvline(umbral, color="gray", linestyle="--",
                linewidth=1, label=f"Cal. mín. {umbral}")
    ax9.set_title("Cal. final vs Asistencia")
    ax9.set_xlabel("Calificación final")
    ax9.set_ylabel("Asistencia")
    ax9.legend(fontsize=8)
    plt.colorbar(scatter, ax=ax9, label="0=Rep / 1=Apr")

    # 2-C  LOO accuracy por alumno (solo modo escaso)
    #       o accuracy del modelo en modo normal
    ax10 = fig.add_subplot(gs[2, 3])
    if scores_loo is not None:
        ax10.plot(range(1, len(scores_loo) + 1), scores_loo,
                  marker="o", markersize=4, color=color_neu, linewidth=1)
        ax10.axhline(scores_loo.mean(), color="tomato", linestyle="--",
                     linewidth=1.2, label=f"Media: {scores_loo.mean():.2%}")
        ax10.set_title("Accuracy LOO por iteración")
        ax10.set_xlabel("Iteración (alumno excluido)")
        ax10.set_ylabel("Accuracy")
        ax10.set_ylim(-0.05, 1.1)
        ax10.legend(fontsize=8)
    else:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        correctos   = int(acc * len(y_test))
        incorrectos = len(y_test) - correctos
        ax10.bar(["Correctas", "Incorrectas"],
                 [correctos, incorrectos],
                 color=[color_ap, color_rep], edgecolor="white", width=0.5)
        ax10.set_title(f"Predicciones — set de prueba\nAccuracy: {acc:.2%}")
        ax10.set_ylabel("Cantidad")
        for i, v in enumerate([correctos, incorrectos]):
            ax10.text(i, v + 0.1, str(v), ha="center",
                      fontsize=12, fontweight="bold")

    nombre_archivo = nombre_grupo.replace(" ", "_").lower()
    plt.savefig(f"{nombre_archivo}_analisis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfica guardada: {nombre_archivo}_analisis.png\n")


# ══════════════════════════════════════════════════════════════════
# SIMULACIÓN
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)

def grupo_aleatorio(n):
    cal = np.random.randint(50, 101, n)
    tar = np.random.randint(30, 101, n)
    asi = np.random.randint(50, 101, n)
    return pd.DataFrame({"cal_final": cal, "prom_tareas": tar, "prom_asist": asi})

model_a, df_a = analizar_grupo(grupo_aleatorio(8),   "Grupo pequeño")
model_b, df_b = analizar_grupo(grupo_aleatorio(35),  "Grupo mediano")
model_c, df_c = analizar_grupo(grupo_aleatorio(120), "Grupo grande")