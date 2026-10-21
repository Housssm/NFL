"""
Tableau de bord des preuves statistiques.

Produit une figure multi-panneaux qui visualise *pourquoi* la conclusion est
atteinte : force des signaux (p-values), effet par domaine (ANOVA), effet net
de l'université (régression contrôlée) et corrélation de rang (Spearman).

Usage :
    python -m src.dashboard               # lit data/output/statistiques.json
    python src/dashboard.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE, "data", "output", "statistiques.json")
SORTIE = os.path.join(BASE, "visualizations", "dashboard_preuves.png")

SEUIL_P = 0.05
VERT = "#2ca02c"
GRIS = "#9e9e9e"
ROUGE = "#d62728"
BLEU = "#1f77b4"


def _mlog10p(p: float) -> float:
    """-log10(p), borné pour l'affichage (p=0 -> très grand)."""
    p = max(float(p), 1e-300)
    return -np.log10(p)


def construire_dashboard(stats: dict, chemin_sortie: str = SORTIE) -> str:
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    _panneau_force_signaux(fig.add_subplot(gs[0, 0]), stats)
    _panneau_anova_domaines(fig.add_subplot(gs[0, 1]), stats)
    _panneau_regression(fig.add_subplot(gs[1, 0]), stats)
    _panneau_spearman(fig.add_subplot(gs[1, 1]), stats)

    fig.suptitle(
        "Tableau de bord des preuves — Université ↔ Réussite professionnelle",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    # Bandeau de conclusion.
    verdict, couleur = _verdict(stats)
    fig.text(
        0.5,
        0.925,
        verdict,
        ha="center",
        va="center",
        fontsize=12.5,
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=couleur, edgecolor="none"),
    )

    fig.savefig(chemin_sortie, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return chemin_sortie


def _verdict(stats: dict) -> tuple[str, str]:
    signaux = 0
    if stats["anova_globale"].get("significatif"):
        signaux += 1
    if stats["regression"].get("effet_universite", {}).get("significatif"):
        signaux += 1
    if stats["chi2"].get("significatif"):
        signaux += 1
    if signaux >= 2:
        return (
            f"CONCLUSION : OUI — corrélation significative "
            f"({signaux}/3 signaux concordants)",
            VERT,
        )
    if signaux == 1:
        return "CONCLUSION : PARTIELLEMENT — 1/3 signal significatif", "#ff7f0e"
    return "CONCLUSION : NON — aucun signal significatif", ROUGE


# --------------------------------------------------------------------------
# Panneau 1 : force des signaux (-log10 p-value)
# --------------------------------------------------------------------------
def _panneau_force_signaux(ax, stats: dict) -> None:
    tests = [
        ("ANOVA globale", stats["anova_globale"]["p_value"]),
        ("Régression\n(effet université)", stats["regression"]["effet_universite"]["p_value"]),
        ("Chi-carré", stats["chi2"]["p_value"]),
        ("Spearman", stats["spearman"]["p_value"]),
    ]
    noms = [t[0] for t in tests]
    valeurs = [_mlog10p(t[1]) for t in tests]
    couleurs = [VERT if t[1] < SEUIL_P else GRIS for t in tests]

    y = np.arange(len(noms))
    ax.barh(y, valeurs, color=couleurs)
    ax.set_yticks(y)
    ax.set_yticklabels(noms, fontsize=10)
    ax.invert_yaxis()

    seuil = _mlog10p(SEUIL_P)
    ax.axvline(seuil, color=ROUGE, linestyle="--", linewidth=1.5)
    ax.text(
        seuil + 1,
        len(noms) - 0.4,
        "seuil p=0.05",
        color=ROUGE,
        fontsize=9,
        rotation=0,
    )
    for yi, (nom, p) in zip(y, tests):
        txt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(
            _mlog10p(p) + 0.5,
            yi,
            txt,
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Force de la preuve  −log₁₀(p-value)  →  plus = plus significatif")
    ax.set_title("1 · Force des signaux statistiques", fontweight="bold", loc="left")
    ax.margins(x=0.18)


# --------------------------------------------------------------------------
# Panneau 2 : ANOVA par domaine
# --------------------------------------------------------------------------
def _panneau_anova_domaines(ax, stats: dict) -> None:
    items = [
        (dom, r["f_statistic"], r["p_value"], r["significatif"])
        for dom, r in stats["anova_par_domaine"].items()
        if r.get("applicable")
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    noms = [i[0] for i in items]
    fstats = [i[1] for i in items]
    couleurs = [VERT if i[3] else GRIS for i in items]

    x = np.arange(len(noms))
    barres = ax.bar(x, fstats, color=couleurs)
    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F-statistic (ANOVA)")
    ax.set_title(
        "2 · L'université compte-t-elle, par domaine ?",
        fontweight="bold",
        loc="left",
    )
    for rect, (_, f, p, sig) in zip(barres, items):
        etoile = "✔" if sig else "✘"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.03,
            etoile,
            ha="center",
            va="bottom",
            fontsize=11,
            color=VERT if sig else ROUGE,
            fontweight="bold",
        )
    # Légende manuelle.
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=VERT, label="significatif (p<0.05)"),
            Patch(facecolor=GRIS, label="non significatif"),
        ],
        fontsize=8,
        loc="upper right",
    )
    ax.margins(y=0.15)


# --------------------------------------------------------------------------
# Panneau 3 : coefficients de régression (effet net université)
# --------------------------------------------------------------------------
def _panneau_regression(ax, stats: dict) -> None:
    coefs = stats["regression"]["coefficients_universite"]
    # Tri croissant pour un diagramme divergent lisible.
    items = sorted(coefs.items(), key=lambda kv: kv[1])
    noms = [k for k, _ in items]
    vals = [v for _, v in items]
    couleurs = [VERT if v > 0 else ROUGE for v in vals]

    y = np.arange(len(noms))
    ax.barh(y, vals, color=couleurs)
    ax.set_yticks(y)
    ax.set_yticklabels(noms, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effet net sur le score (points) — à expérience, domaine\net taille d'entreprise égaux")
    r2 = stats["regression"]["r_squared"]
    fe = stats["regression"]["effet_universite"]
    ax.set_title(
        f"3 · Effet net de l'université (régression contrôlée)\n"
        f"R²={r2:.2f} · effet université : F={fe['F']:.1f}, p<0.001",
        fontweight="bold",
        loc="left",
        fontsize=10.5,
    )


# --------------------------------------------------------------------------
# Panneau 4 : Spearman rang vs score moyen
# --------------------------------------------------------------------------
def _panneau_spearman(ax, stats: dict) -> None:
    classement = stats["spearman"]["classement_universites"]
    # classement : université -> score moyen, déjà trié décroissant.
    universites = list(classement.keys())
    scores = list(classement.values())
    n = len(universites)
    # Rang affiché : 1 = meilleure.
    rangs = np.arange(1, n + 1)

    ax.scatter(rangs, scores, color=BLEU, s=55, zorder=3)
    # Ligne de tendance.
    coef = np.polyfit(rangs, scores, 1)
    xs = np.linspace(1, n, 100)
    ax.plot(xs, np.polyval(coef, xs), "r--", linewidth=1.8, zorder=2)

    # Annoter le top 3 et le bottom 2.
    for i in list(range(3)) + [n - 2, n - 1]:
        ax.annotate(
            universites[i].replace("Université ", "U. "),
            (rangs[i], scores[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
        )
    rho = stats["spearman"]["rho"]
    ax.set_xlabel("Rang de l'université (1 = meilleure)")
    ax.set_ylabel("Score de réussite moyen")
    ax.set_title(
        f"4 · Corrélation de rang (Spearman ρ={rho:.2f}, p<0.001)",
        fontweight="bold",
        loc="left",
    )
    ax.invert_xaxis()  # rang 1 à droite -> meilleur score en haut à droite


def main() -> str:
    with open(JSON_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload["statistiques"]
    chemin = construire_dashboard(stats)
    print(f"Dashboard généré -> {chemin}")
    return chemin


if __name__ == "__main__":
    main()
