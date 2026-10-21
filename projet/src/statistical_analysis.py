"""
Tests statistiques pour mesurer l'effet de l'université sur la réussite.

Contient :
  - ANOVA (par domaine) : différence de moyennes entre universités
  - Chi-carré : indépendance université / catégorie de réussite
  - Corrélation de Spearman : rang université vs score
  - Régression linéaire multiple : effet université en contrôlant covariables
  - Classements par domaine avec IC 95% et Cohen's d
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

ALPHA = 0.05


# --------------------------------------------------------------------------
# A. ANOVA par domaine
# --------------------------------------------------------------------------
def anova_par_domaine(df: pd.DataFrame, min_groupes: int = 2, min_n: int = 2) -> dict:
    """Pour chaque domaine, ANOVA du score global entre universités."""
    resultats = {}
    for domaine, sous in df.groupby("domaine_professionnel"):
        groupes = [
            g["score_global"].values
            for _, g in sous.groupby("universite")
            if len(g) >= min_n
        ]
        if len(groupes) < min_groupes:
            resultats[domaine] = {
                "applicable": False,
                "raison": "Pas assez d'universités avec données suffisantes",
                "n_universites": len(groupes),
            }
            continue
        f_stat, p_value = stats.f_oneway(*groupes)
        resultats[domaine] = {
            "applicable": True,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significatif": bool(p_value < ALPHA),
            "n_universites": len(groupes),
            "n_observations": int(sum(len(g) for g in groupes)),
        }
    return resultats


def anova_globale(df: pd.DataFrame) -> dict:
    """ANOVA globale (toutes universités confondues, tous domaines)."""
    groupes = [
        g["score_global"].values
        for _, g in df.groupby("universite")
        if len(g) >= 2
    ]
    if len(groupes) < 2:
        return {"applicable": False}
    f_stat, p_value = stats.f_oneway(*groupes)
    return {
        "applicable": True,
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significatif": bool(p_value < ALPHA),
        "n_universites": len(groupes),
    }


# --------------------------------------------------------------------------
# B. Chi-carré : université vs catégorie de réussite
# --------------------------------------------------------------------------
def chi2_independance(df: pd.DataFrame) -> dict:
    if "categorie_reussite" not in df.columns:
        return {"applicable": False, "raison": "categorie_reussite absente"}
    table = pd.crosstab(df["universite"], df["categorie_reussite"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"applicable": False, "raison": "Table de contingence trop petite"}

    chi2, p, dof, attendu = stats.chi2_contingency(table)
    # V de Cramér comme taille d'effet.
    n = table.values.sum()
    k = min(table.shape) - 1
    cramer_v = float(np.sqrt(chi2 / (n * k))) if n and k else float("nan")
    return {
        "applicable": True,
        "chi2": float(chi2),
        "p_value": float(p),
        "degres_liberte": int(dof),
        "significatif": bool(p < ALPHA),
        "cramer_v": cramer_v,
        "table": table,
    }


# --------------------------------------------------------------------------
# C. Corrélation de Spearman : rang université vs score
# --------------------------------------------------------------------------
def spearman_rang_universite(df: pd.DataFrame) -> dict:
    """Corrèle le rang de l'université (par score moyen) au score individuel.

    On ordonne les universités par score moyen, on assigne un rang, puis on
    mesure la corrélation de Spearman entre ce rang et le score individuel.
    """
    moyennes = (
        df.groupby("universite")["score_global"].mean().sort_values(ascending=False)
    )
    n_univ = len(moyennes)
    # Rang ordinal : meilleure université = rang le plus élevé, afin qu'une
    # corrélation positive signifie « mieux classé ↔ score plus élevé ».
    rang = {u: n_univ - i for i, u in enumerate(moyennes.index)}
    rangs_individuels = df["universite"].map(rang).astype(float)

    rho, p = stats.spearmanr(rangs_individuels, df["score_global"])
    return {
        "applicable": True,
        "rho": float(rho),
        "p_value": float(p),
        "significatif": bool(p < ALPHA),
        "classement_universites": moyennes.round(2).to_dict(),
    }


# --------------------------------------------------------------------------
# D. Régression linéaire multiple
# --------------------------------------------------------------------------
def regression_multiple(df: pd.DataFrame) -> dict:
    """Isole l'effet de l'université en contrôlant expérience, domaine, taille.

    Modèle : score_global ~ C(universite) + annees_experience
             + C(domaine_professionnel) + log(taille_entreprise)
    """
    data = df.copy()
    data["log_taille"] = np.log1p(data["taille_entreprise"].clip(lower=1))

    formule = (
        "score_global ~ C(universite) + annees_experience "
        "+ C(domaine_professionnel) + log_taille"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modele = smf.ols(formule, data=data).fit()

    # Significativité globale du facteur université via test F (type II).
    try:
        anova_tab = sm.stats.anova_lm(modele, typ=2)
        ligne_univ = anova_tab.loc["C(universite)"]
        effet_univ = {
            "F": float(ligne_univ["F"]),
            "p_value": float(ligne_univ["PR(>F)"]),
            "significatif": bool(ligne_univ["PR(>F)"] < ALPHA),
        }
    except Exception as exc:  # pragma: no cover - robustesse
        effet_univ = {"erreur": str(exc)}

    # Coefficients des universités (écart au niveau de référence).
    coefs = {
        nom.replace("C(universite)[T.", "").rstrip("]"): float(val)
        for nom, val in modele.params.items()
        if nom.startswith("C(universite)")
    }

    return {
        "applicable": True,
        "r_squared": float(modele.rsquared),
        "r_squared_adj": float(modele.rsquared_adj),
        "f_statistic": float(modele.fvalue),
        "f_pvalue": float(modele.f_pvalue),
        "n_observations": int(modele.nobs),
        "effet_universite": effet_univ,
        "coefficients_universite": coefs,
        "summary_text": modele.summary().as_text(),
    }


# --------------------------------------------------------------------------
# E. Classements par domaine + IC 95% + Cohen's d
# --------------------------------------------------------------------------
def _ic95(serie: pd.Series) -> tuple[float, float, float]:
    """Retourne (moyenne, demi_largeur_IC, erreur_standard)."""
    n = len(serie)
    moyenne = float(serie.mean())
    if n < 2:
        return moyenne, float("nan"), float("nan")
    se = float(serie.std(ddof=1) / np.sqrt(n))
    t = stats.t.ppf(0.975, df=n - 1)
    return moyenne, t * se, se


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d entre deux échantillons (pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def classement_par_domaine(df: pd.DataFrame, min_n: int = 2) -> dict:
    """Classe les universités par domaine avec IC 95%, p-value et Cohen's d.

    Pour chaque université, un test t compare ses scores au reste du domaine
    (one-vs-rest) ; Cohen's d mesure l'écart de la meilleure à la moyenne.
    """
    resultats = {}
    for domaine, sous in df.groupby("domaine_professionnel"):
        lignes = []
        # Seuil du top 25% salarial, calculé sur le domaine.
        seuil_top25_salaire = sous["salaire_annuel"].quantile(0.75)
        for univ, g in sous.groupby("universite"):
            if len(g) < min_n:
                continue
            moyenne, demi_ic, se = _ic95(g["score_global"])
            reste = sous[sous["universite"] != univ]["score_global"]
            if len(reste) >= 2 and len(g) >= 2:
                t_stat, p_val = stats.ttest_ind(
                    g["score_global"], reste, equal_var=False
                )
                d = _cohen_d(g["score_global"], reste)
            else:
                t_stat, p_val, d = float("nan"), float("nan"), float("nan")
            lignes.append(
                {
                    "universite": univ,
                    "n": int(len(g)),
                    "score_moyen": round(moyenne, 2),
                    "ic95_demi_largeur": round(demi_ic, 2)
                    if not np.isnan(demi_ic)
                    else None,
                    "ic95_bas": round(moyenne - demi_ic, 2)
                    if not np.isnan(demi_ic)
                    else None,
                    "ic95_haut": round(moyenne + demi_ic, 2)
                    if not np.isnan(demi_ic)
                    else None,
                    "salaire_median": float(g["salaire_annuel"].median()),
                    "taux_top25": round(
                        float(
                            (g["salaire_annuel"] >= seuil_top25_salaire).mean()
                        )
                        * 100,
                        1,
                    ),
                    "p_value_vs_reste": round(float(p_val), 4)
                    if not np.isnan(p_val)
                    else None,
                    "significatif": bool(p_val < ALPHA)
                    if not np.isnan(p_val)
                    else False,
                    "cohen_d_vs_reste": round(float(d), 3)
                    if not np.isnan(d)
                    else None,
                }
            )
        classement = sorted(lignes, key=lambda x: x["score_moyen"], reverse=True)
        resultats[domaine] = classement
    return resultats


def interpreter_force(valeur: float, type_: str = "r") -> str:
    """Interprète une taille d'effet / corrélation en français."""
    v = abs(valeur)
    if type_ in ("r", "rho", "cramer"):
        if v < 0.1:
            return "négligeable"
        if v < 0.3:
            return "faible"
        if v < 0.5:
            return "modérée"
        return "forte"
    if type_ == "d":  # Cohen's d
        if v < 0.2:
            return "négligeable"
        if v < 0.5:
            return "petite"
        if v < 0.8:
            return "moyenne"
        return "grande"
    return "indéterminée"


def analyser(df: pd.DataFrame) -> dict:
    """Exécute l'ensemble des analyses et retourne un dictionnaire."""
    return {
        "anova_globale": anova_globale(df),
        "anova_par_domaine": anova_par_domaine(df),
        "chi2": chi2_independance(df),
        "spearman": spearman_rang_universite(df),
        "regression": regression_multiple(df),
        "classement_par_domaine": classement_par_domaine(df),
    }
