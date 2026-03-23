from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ══════════════════════════════════════════════════════════════════
# PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════
def preparar_datos(df: pd.DataFrame, umbral: int):
    n  = len(df)
    df = df.copy()
    df["aprueba"] = (df["cal_final"] >= umbral).astype(int)

    X = df[["cal_final", "prom_tareas", "prom_asist"]]
    y = df["aprueba"]
    _, conteos = np.unique(y, return_counts=True)

    if n < 15:
        return df, X, y, None, None, None, None, \
               f"Datos escasos ({n} alumnos) — LOO CV", True

    min_ratio = conteos.min() / n
    if min_ratio < 0.20:
        df_maj = df[y == conteos.argmax()]
        df_min = df[y == conteos.argmin()]
        df_up  = resample(df_min, replace=True,
                          n_samples=len(df_maj), random_state=42)
        df_bal = pd.concat([df_maj, df_up])
        X = df_bal[["cal_final", "prom_tareas", "prom_asist"]]
        y = df_bal["aprueba"]
        modo = f"Desbalance ({min_ratio:.0%}) — Oversampling"
    else:
        modo = f"Normal ({n} alumnos) — Train/Test split"

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, modo, False


# ══════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════
def entrenar(X, y, X_train, X_test, y_train, y_test, loo_mode: bool):
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=42)
    if loo_mode:
        scores = cross_val_score(model, X, y,
                                 cv=LeaveOneOut(), scoring="accuracy")
        model.fit(X, y)
        return {
            "model":      model,
            "scores_loo": scores,
            "y_pred":     model.predict(X),
            "y_prob":     model.predict_proba(X)[:, 1],
            "y_test":     y,
            "X_test":     X,
        }

    model.fit(X_train, y_train)
    return {
        "model":      model,
        "scores_loo": None,
        "y_pred":     model.predict(X_test),
        "y_prob":     model.predict_proba(X_test)[:, 1],
        "y_test":     y_test,
        "X_test":     X_test,
    }


# ══════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════
def graficar(df, res, nombre_grupo, modo, umbral):
    n         = len(df)
    color_m   = "mediumseagreen"
    color_ap  = "#4CAF50"
    color_rep = "#F44336"

    model      = res["model"]
    y_pred     = np.array(res["y_pred"])
    y_prob     = np.array(res["y_prob"])
    y_test_arr = np.array(res["y_test"])
    scores_loo = res["scores_loo"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{nombre_grupo}  ·  Random Forest  ·  {modo}",
                 fontsize=13, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # ── Fila 0: panorama del grupo ────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["cal_final"], bins=12, color=color_m,
            edgecolor="white", linewidth=0.5)
    ax.axvline(umbral, color="orange", linestyle="--",
               linewidth=1.5, label=f"Mínimo ({umbral})")
    ax.axvline(df["cal_final"].mean(), color="tomato", linestyle=":",
               linewidth=1.5, label=f"Media ({df['cal_final'].mean():.1f})")
    ax.set_title("Dist. calificación final")
    ax.set_xlabel("Calificación"); ax.set_ylabel("Alumnos")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    counts = df["aprueba"].value_counts().sort_index()
    ax.pie(counts, labels=["Reprueba", "Aprueba"], autopct="%1.1f%%",
           colors=[color_rep, color_ap], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    ax.set_title("Aprueba / Reprueba")

    ax = fig.add_subplot(gs[0, 2])
    df_melt = df.melt(id_vars="aprueba",
                      value_vars=["cal_final", "prom_tareas", "prom_asist"],
                      var_name="variable", value_name="valor")
    df_melt["variable"] = df_melt["variable"].map({
        "cal_final": "Cal.", "prom_tareas": "Tareas", "prom_asist": "Asist."})
    df_melt["resultado"] = df_melt["aprueba"].map({1: "Aprueba", 0: "Reprueba"})
    sns.boxplot(data=df_melt, x="variable", y="valor", hue="resultado",
                palette={"Aprueba": color_ap, "Reprueba": color_rep},
                ax=ax, linewidth=0.8)
    ax.set_title("Variables por resultado")
    ax.set_xlabel(""); ax.legend(title="", fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    corr = df[["cal_final", "prom_tareas", "prom_asist", "aprueba"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.4, cbar=False,
                xticklabels=["Cal.", "Tareas", "Asist.", "Apr."],
                yticklabels=["Cal.", "Tareas", "Asist.", "Apr."],
                annot_kws={"size": 8})
    ax.set_title("Correlación")

    # ── Fila 1: métricas del modelo ───────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test_arr, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                linewidths=0.5,
                xticklabels=["Reprueba", "Aprueba"],
                yticklabels=["Reprueba", "Aprueba"])
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(y_prob[y_test_arr == 0], bins=10, alpha=0.7,
            color=color_rep, label="Reprueba real")
    ax.hist(y_prob[y_test_arr == 1], bins=10, alpha=0.7,
            color=color_ap,  label="Aprueba real")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Distribución P(aprueba)")
    ax.set_xlabel("Probabilidad"); ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    if len(np.unique(y_test_arr)) == 2:
        fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color_m, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.1, color=color_m)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Una sola clase\nen prueba",
                ha="center", va="center", color="gray")
    ax.set_title("Curva ROC")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")

    # Importancia de variables — característica distintiva de Random Forest
    ax = fig.add_subplot(gs[1, 3])
    features     = ["Cal. final", "Tareas", "Asistencia"]
    importancias = model.feature_importances_
    std          = np.std([t.feature_importances_ for t in model.estimators_], axis=0)
    ax.barh(features, importancias, xerr=std, color=color_m,
            edgecolor="white", capsize=4)
    ax.set_xlabel("Importancia (± std entre árboles)")
    ax.set_title("Importancia de variables")
    for i, v in enumerate(importancias):
        ax.text(v + std[i] + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    # ── Fila 2: comportamiento individual ─────────────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    all_prob = model.predict_proba(
        df[["cal_final", "prom_tareas", "prom_asist"]])[:, 1]
    idx_sort = np.argsort(all_prob)
    bar_cols = [color_ap if df["aprueba"].values[i] == 1
                else color_rep for i in idx_sort]
    ax.barh(range(n), all_prob[idx_sort], color=bar_cols,
            edgecolor="white", height=0.7)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("P(aprueba) por alumno — grupo completo\n"
                 "(verde=aprueba real, rojo=reprueba real)")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_yticks([]); ax.set_xlim(0, 1.05)

    ax = fig.add_subplot(gs[2, 2])
    scatter = ax.scatter(
        df["cal_final"], df["prom_asist"],
        c=df["aprueba"], cmap="RdYlGn",
        edgecolors="k", linewidths=0.3, s=45, alpha=0.85)
    ax.axvline(umbral, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Cal. final vs Asistencia")
    ax.set_xlabel("Calificación final"); ax.set_ylabel("Asistencia")
    plt.colorbar(scatter, ax=ax, label="0=Rep / 1=Apr")

    ax = fig.add_subplot(gs[2, 3])
    if scores_loo is not None:
        ax.plot(range(1, len(scores_loo) + 1), scores_loo,
                marker="o", markersize=4, color=color_m, linewidth=1)
        ax.axhline(scores_loo.mean(), color="tomato", linestyle="--",
                   linewidth=1.2, label=f"Media: {scores_loo.mean():.2%}")
        ax.set_title("Accuracy LOO por iteración")
        ax.set_xlabel("Iteración"); ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=8)
    else:
        acc         = accuracy_score(y_test_arr, y_pred)
        correctos   = int((y_pred == y_test_arr).sum())
        incorrectos = len(y_test_arr) - correctos
        ax.bar(["Correctas", "Incorrectas"],
               [correctos, incorrectos],
               color=[color_ap, color_rep], edgecolor="white", width=0.5)
        ax.set_title(f"Predicciones — set de prueba\nAccuracy: {acc:.2%}")
        ax.set_ylabel("Cantidad")
        for i, v in enumerate([correctos, incorrectos]):
            ax.text(i, v + 0.1, str(v), ha="center",
                    fontsize=12, fontweight="bold")

    slug = f"{nombre_grupo}_random_forest".replace(" ", "_").lower()
    plt.savefig(f"{slug}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {slug}.png\n")


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def analizar_grupo(df: pd.DataFrame,
                   nombre_grupo: str = "Grupo",
                   umbral_aprobatorio: int = 70):

    df, X, y, X_train, X_test, y_train, y_test, modo, loo_mode = \
        preparar_datos(df, umbral_aprobatorio)

    res = entrenar(X, y, X_train, X_test, y_train, y_test, loo_mode)
    graficar(df, res, nombre_grupo, modo, umbral_aprobatorio)

    print(classification_report(res["y_test"], res["y_pred"],
          target_names=["Reprueba", "Aprueba"], zero_division=0))

    return res


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