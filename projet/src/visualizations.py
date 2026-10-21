"""
Génération des visualisations (PNG) via Matplotlib / Seaborn.

Backend non interactif ('Agg') pour fonctionner sans affichage.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def _sauver(fig, chemin: str) -> str:
    fig.savefig(chemin)
    plt.close(fig)
    return chemin


def heatmap_scores(df: pd.DataFrame, dossier: str) -> str:
    """Heatmap Université (lignes) × Domaine (colonnes) = score moyen."""
    pivot = df.pivot_table(
        index="universite",
        columns="domaine_professionnel",
        values="score_global",
        aggfunc="mean",
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    hauteur = max(6, 0.45 * len(pivot))
    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 4, hauteur))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Score de réussite moyen"},
        ax=ax,
    )
    ax.set_title("Score de réussite moyen par université et domaine")
    ax.set_xlabel("Domaine professionnel")
    ax.set_ylabel("Université")
    return _sauver(fig, os.path.join(dossier, "heatmap_scores.png"))


def boxplot_top_universites(df: pd.DataFrame, dossier: str, top: int = 10) -> str:
    """Box plots de la distribution des scores pour le top N universités."""
    ordre = (
        df.groupby("universite")["score_global"]
        .mean()
        .sort_values(ascending=False)
        .head(top)
        .index.tolist()
    )
    sous = df[df["universite"].isin(ordre)]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=sous,
        x="score_global",
        y="universite",
        order=ordre,
        hue="universite",
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set_title(f"Distribution des scores — Top {len(ordre)} universités")
    ax.set_xlabel("Score de réussite global")
    ax.set_ylabel("Université")
    return _sauver(fig, os.path.join(dossier, "boxplot_top_universites.png"))


def barres_top_par_domaine(
    classement: dict, dossier: str, top: int = 15
) -> list[str]:
    """Bar chart Top N universités par domaine avec barres d'erreur (IC95)."""
    chemins = []
    for domaine, lignes in classement.items():
        if not lignes:
            continue
        sous = lignes[:top]
        noms = [l["universite"] for l in sous]
        scores = [l["score_moyen"] for l in sous]
        erreurs = [
            l["ic95_demi_largeur"] if l["ic95_demi_largeur"] is not None else 0
            for l in sous
        ]

        fig, ax = plt.subplots(figsize=(11, max(5, 0.5 * len(noms) + 2)))
        y = np.arange(len(noms))
        couleurs = [
            "#2ca02c" if l["significatif"] else "#7f7f7f" for l in sous
        ]
        ax.barh(y, scores, xerr=erreurs, color=couleurs, capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(noms)
        ax.invert_yaxis()
        ax.set_xlabel("Score de réussite moyen (± IC 95%)")
        ax.set_title(f"Top universités — {domaine}\n(vert = significatif vs reste)")
        chemin = os.path.join(
            dossier, f"top_universites_{_slug(domaine)}.png"
        )
        chemins.append(_sauver(fig, chemin))
    return chemins


def scatter_score_salaire(df: pd.DataFrame, dossier: str) -> str:
    """Nuage de points score composite vs salaire, coloré par domaine."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="salaire_annuel",
        y="score_global",
        hue="domaine_professionnel",
        alpha=0.5,
        ax=ax,
    )
    # Ligne de tendance globale.
    coef = np.polyfit(df["salaire_annuel"], df["score_global"], 1)
    xs = np.linspace(df["salaire_annuel"].min(), df["salaire_annuel"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), "r--", linewidth=2, label="Tendance")
    r = df["salaire_annuel"].corr(df["score_global"])
    ax.set_title(f"Score composite vs salaire annuel (r = {r:.2f})")
    ax.set_xlabel("Salaire annuel (€)")
    ax.set_ylabel("Score de réussite global")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _sauver(fig, os.path.join(dossier, "scatter_score_salaire.png"))


def histogrammes_scores(df: pd.DataFrame, dossier: str, top: int = 6) -> str:
    """Histogrammes des scores pour les principales universités (facettes)."""
    ordre = (
        df.groupby("universite")["score_global"]
        .mean()
        .sort_values(ascending=False)
        .head(top)
        .index.tolist()
    )
    sous = df[df["universite"].isin(ordre)]
    g = sns.FacetGrid(
        sous,
        col="universite",
        col_wrap=3,
        col_order=ordre,
        height=3,
        sharex=True,
        sharey=False,
    )
    g.map_dataframe(sns.histplot, x="score_global", bins=15, kde=True, color="#4c72b0")
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Score global", "Effectif")
    g.figure.suptitle("Distribution des scores par université (top)", y=1.02)
    chemin = os.path.join(dossier, "histogrammes_scores.png")
    g.figure.savefig(chemin)
    plt.close(g.figure)
    return chemin


def matrice_correlation(df: pd.DataFrame, dossier: str) -> str:
    """Matrice de corrélation entre variables numériques."""
    cols = [
        "annees_experience",
        "salaire_annuel",
        "niveau_poste",
        "taille_entreprise",
        "score_salaire",
        "score_niveau",
        "score_vitesse",
        "score_global",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matrice de corrélation des variables numériques")
    return _sauver(fig, os.path.join(dossier, "matrice_correlation.png"))


def _slug(texte: str) -> str:
    import re
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", texte)
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s


def generer_toutes(
    df: pd.DataFrame, classement: dict, dossier: str
) -> list[str]:
    """Génère toutes les visualisations et retourne la liste des chemins."""
    os.makedirs(dossier, exist_ok=True)
    chemins: list[str] = []
    chemins.append(heatmap_scores(df, dossier))
    chemins.append(boxplot_top_universites(df, dossier))
    chemins.extend(barres_top_par_domaine(classement, dossier))
    chemins.append(scatter_score_salaire(df, dossier))
    chemins.append(histogrammes_scores(df, dossier))
    chemins.append(matrice_correlation(df, dossier))
    return chemins
