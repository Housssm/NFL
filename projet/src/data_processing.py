"""
Nettoyage et préparation des données.

- Normalisation des noms d'universités (casse, accents, abréviations)
- Gestion des valeurs manquantes
- Détection et suppression des doublons
- Traitement des outliers (salaires aberrants) via IQR
- Filtrage des universités sous-représentées
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

COLONNES_REQUISES = [
    "id",
    "universite",
    "annee_diplome",
    "domaine_professionnel",
    "annees_experience",
    "salaire_annuel",
    "niveau_poste",
    "taille_entreprise",
]

# Table de normalisation : variante normalisée -> nom canonique.
# La clé est calculée par `_cle_normalisation`.
ALIAS_CANONIQUES = {
    "universite paris-saclay": "Université Paris-Saclay",
    "ecole polytechnique": "École Polytechnique",
    "sorbonne universite": "Sorbonne Université",
    "universite psl": "Université PSL",
    "universite grenoble alpes": "Université Grenoble Alpes",
    "universite de strasbourg": "Université de Strasbourg",
    "universite de lyon": "Université de Lyon",
    "universite d'aix-marseille": "Université d'Aix-Marseille",
    "universite de bordeaux": "Université de Bordeaux",
    "universite de lille": "Université de Lille",
    "universite de montpellier": "Université de Montpellier",
    "universite de nantes": "Université de Nantes",
    "universite de rennes": "Université de Rennes",
    "universite toulouse capitole": "Université Toulouse Capitole",
    "universite cote d'azur": "Université Côte d'Azur",
    "universite de lorraine": "Université de Lorraine",
    "universite paris cite": "Université Paris Cité",
    "universite de bourgogne": "Université de Bourgogne",
}


def _cle_normalisation(nom: str) -> str:
    """Produit une clé insensible à la casse et aux accents."""
    if not isinstance(nom, str):
        return ""
    # Retrait des accents.
    nfkd = unicodedata.normalize("NFKD", nom)
    sans_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    sans_accent = sans_accent.lower().strip()
    # Espaces multiples -> simple.
    sans_accent = re.sub(r"\s+", " ", sans_accent)
    return sans_accent


def normaliser_universites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cles = df["universite"].map(_cle_normalisation)
    df["universite"] = cles.map(ALIAS_CANONIQUES).fillna(
        df["universite"].astype(str).str.strip()
    )
    return df


def valider_colonnes(df: pd.DataFrame) -> None:
    manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes requises manquantes dans le CSV : {manquantes}"
        )


def supprimer_doublons(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    avant = len(df)
    # On ignore la colonne id pour détecter les vrais doublons de contenu.
    cols = [c for c in df.columns if c != "id"]
    df = df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    return df, avant - len(df)


def gerer_valeurs_manquantes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    rapport = {}

    # Lignes sans salaire : imputées par la médiane du domaine
    # (le salaire est central dans le score, mais on évite de jeter la ligne).
    n_sal_nan = int(df["salaire_annuel"].isna().sum())
    if n_sal_nan:
        df["salaire_annuel"] = df.groupby("domaine_professionnel")[
            "salaire_annuel"
        ].transform(lambda s: s.fillna(s.median()))
        # S'il reste des NaN (domaine entièrement vide), médiane globale.
        df["salaire_annuel"] = df["salaire_annuel"].fillna(
            df["salaire_annuel"].median()
        )
    rapport["salaires_imputes"] = n_sal_nan

    # Satisfaction : colonne optionnelle, imputée par la médiane globale.
    if "satisfaction_carriere" in df.columns:
        n_sat_nan = int(df["satisfaction_carriere"].isna().sum())
        df["satisfaction_carriere"] = df["satisfaction_carriere"].fillna(
            df["satisfaction_carriere"].median()
        )
        rapport["satisfaction_imputee"] = n_sat_nan

    # Lignes sans donnée structurante indispensable : supprimées.
    essentielles = [
        "universite",
        "domaine_professionnel",
        "annees_experience",
        "niveau_poste",
    ]
    avant = len(df)
    df = df.dropna(subset=essentielles).reset_index(drop=True)
    rapport["lignes_supprimees_na"] = avant - len(df)

    return df, rapport


def traiter_outliers(df: pd.DataFrame, colonne: str = "salaire_annuel") -> tuple[pd.DataFrame, int]:
    """Winsorisation par domaine via la règle IQR (1.5*IQR).

    On borne plutôt que de supprimer, pour conserver la taille d'échantillon.
    """
    df = df.copy()
    n_modifies = 0

    def _winsor(groupe: pd.Series) -> pd.Series:
        nonlocal n_modifies
        q1, q3 = groupe.quantile(0.25), groupe.quantile(0.75)
        iqr = q3 - q1
        bas, haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        hors = ((groupe < bas) | (groupe > haut)).sum()
        n_modifies += int(hors)
        return groupe.clip(lower=bas, upper=haut)

    df[colonne] = df.groupby("domaine_professionnel")[colonne].transform(_winsor)
    return df, n_modifies


def filtrer_universites(
    df: pd.DataFrame, min_observations: int = 30
) -> tuple[pd.DataFrame, list[str]]:
    counts = df["universite"].value_counts()
    gardees = counts[counts >= min_observations].index.tolist()
    exclues = counts[counts < min_observations].index.tolist()
    df = df[df["universite"].isin(gardees)].reset_index(drop=True)
    return df, exclues


def nettoyer(
    df: pd.DataFrame, min_observations: int = 30
) -> tuple[pd.DataFrame, dict]:
    """Pipeline complet de nettoyage. Retourne (df_propre, rapport)."""
    valider_colonnes(df)
    rapport: dict = {"lignes_initiales": len(df)}

    df = normaliser_universites(df)
    df, n_dups = supprimer_doublons(df)
    rapport["doublons_supprimes"] = n_dups

    df, rapport_na = gerer_valeurs_manquantes(df)
    rapport.update(rapport_na)

    # Typage numérique robuste.
    for col in [
        "annees_experience",
        "salaire_annuel",
        "niveau_poste",
        "taille_entreprise",
        "annee_diplome",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["salaire_annuel", "niveau_poste"]).reset_index(drop=True)

    # annees_experience ne doit pas être nulle pour les scores de progression.
    df["annees_experience"] = df["annees_experience"].clip(lower=0)

    df, n_out = traiter_outliers(df)
    rapport["outliers_winsorises"] = n_out

    df, exclues = filtrer_universites(df, min_observations)
    rapport["universites_exclues"] = exclues
    rapport["min_observations"] = min_observations
    rapport["lignes_finales"] = len(df)
    rapport["universites_retenues"] = sorted(df["universite"].unique().tolist())
    rapport["domaines"] = sorted(df["domaine_professionnel"].unique().tolist())

    return df, rapport
