"""
Calcul des scores de réussite professionnelle.

Score composite = moyenne pondérée de :
  - Score salarial    (40%) : percentile du salaire dans son domaine
  - Score hiérarchique(35%) : niveau de poste normalisé (1..6 -> 0..1)
  - Score de vitesse  (25%) : rapidité d'accession à un poste senior/manager

Tous les sous-scores sont ramenés sur une échelle 0..100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PONDERATIONS = {"salaire": 0.40, "niveau": 0.35, "vitesse": 0.25}

# Niveau considéré comme "senior/manager" (seuil de réussite de carrière).
SEUIL_SENIOR = 3  # 3 = Senior


def _percentile_par_domaine(df: pd.DataFrame, colonne: str) -> pd.Series:
    """Rang en percentile (0..100) à l'intérieur de chaque domaine."""
    return (
        df.groupby("domaine_professionnel")[colonne]
        .rank(pct=True)
        .mul(100.0)
    )


def calculer_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes de scores au DataFrame et le retourne."""
    df = df.copy()

    # --- Score salarial : percentile intra-domaine --------------------
    df["score_salaire"] = _percentile_par_domaine(df, "salaire_annuel")

    # --- Score hiérarchique : niveau normalisé 1..6 -> 0..100 ---------
    df["score_niveau"] = (df["niveau_poste"] - 1) / (6 - 1) * 100.0

    # --- Score de vitesse de progression ------------------------------
    # Progression = niveau atteint par année d'expérience.
    # On évite la division par zéro : exp effective = max(exp, 1).
    exp_eff = df["annees_experience"].clip(lower=1)
    progression = df["niveau_poste"] / exp_eff
    # Normalisation en percentile intra-domaine pour comparer équitablement.
    df["_progression"] = progression
    df["score_vitesse"] = _percentile_par_domaine(df, "_progression")
    df = df.drop(columns="_progression")

    # --- Indicateur : a atteint senior/manager ------------------------
    df["atteint_senior"] = (df["niveau_poste"] >= SEUIL_SENIOR).astype(int)

    # --- Score global pondéré -----------------------------------------
    df["score_global"] = (
        PONDERATIONS["salaire"] * df["score_salaire"]
        + PONDERATIONS["niveau"] * df["score_niveau"]
        + PONDERATIONS["vitesse"] * df["score_vitesse"]
    )

    return df


def categoriser_reussite(df: pd.DataFrame, n_categories: int = 3) -> pd.DataFrame:
    """Catégorise le score global (pour le test du Chi-carré).

    3 catégories par défaut : Faible / Moyen / Élevé via terciles.
    """
    df = df.copy()
    labels = {2: ["Faible", "Élevé"], 3: ["Faible", "Moyen", "Élevé"]}.get(
        n_categories, [f"Q{i+1}" for i in range(n_categories)]
    )
    try:
        df["categorie_reussite"] = pd.qcut(
            df["score_global"], q=n_categories, labels=labels, duplicates="drop"
        )
    except ValueError:
        # Trop peu de valeurs distinctes : repli sur un découpage simple.
        df["categorie_reussite"] = pd.cut(
            df["score_global"], bins=n_categories, labels=labels
        )
    return df


def resume_par_universite_domaine(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques descriptives par université × domaine professionnel."""
    # Seuil du top 25% salarial, calculé par domaine.
    seuils_top25 = df.groupby("domaine_professionnel")["salaire_annuel"].transform(
        lambda s: s.quantile(0.75)
    )
    df = df.copy()
    df["_top25_salaire"] = (df["salaire_annuel"] >= seuils_top25).astype(int)

    grp = df.groupby(["universite", "domaine_professionnel"], observed=True)
    resume = grp.agg(
        n_diplomes=("id", "count"),
        score_moyen=("score_global", "mean"),
        score_ecart_type=("score_global", "std"),
        salaire_median=("salaire_annuel", "median"),
        salaire_moyen=("salaire_annuel", "mean"),
        niveau_moyen=("niveau_poste", "mean"),
        taux_top25_salaire=("_top25_salaire", "mean"),
        taux_senior=("atteint_senior", "mean"),
    ).reset_index()

    resume["taux_top25_salaire"] = (resume["taux_top25_salaire"] * 100).round(1)
    resume["taux_senior"] = (resume["taux_senior"] * 100).round(1)
    resume = resume.round(2)
    return resume


def distribution_niveaux(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution (comptes) des niveaux hiérarchiques par université."""
    dist = (
        df.groupby(["universite", "niveau_poste"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    dist.columns = [f"niveau_{int(c)}" for c in dist.columns]
    return dist.reset_index()
