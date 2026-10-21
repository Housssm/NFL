"""
Génération d'un jeu de données synthétique réaliste.

Produit des diplômés de plusieurs universités françaises réparties sur
plusieurs domaines professionnels, avec une corrélation volontairement
introduite pour certaines universités "d'élite" afin de pouvoir tester
la détection statistique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Référentiels ---------------------------------------------------------

# Universités françaises réelles. Le "prestige_base" (0..1) sert uniquement
# à injecter une corrélation synthétique ; il n'apparaît pas dans le CSV.
UNIVERSITES = {
    "Université Paris-Saclay": 0.92,
    "École Polytechnique": 0.95,
    "Sorbonne Université": 0.80,
    "Université PSL": 0.90,
    "Université Grenoble Alpes": 0.70,
    "Université de Strasbourg": 0.65,
    "Université de Lyon": 0.68,
    "Université d'Aix-Marseille": 0.60,
    "Université de Bordeaux": 0.63,
    "Université de Lille": 0.58,
    "Université de Montpellier": 0.57,
    "Université de Nantes": 0.55,
    "Université de Rennes": 0.56,
    "Université Toulouse Capitole": 0.66,
    "Université Côte d'Azur": 0.59,
    "Université de Lorraine": 0.52,
    "Université Paris Cité": 0.78,
    "Université de Bourgogne": 0.50,
}

# Domaines professionnels et leur médiane salariale de base (en €).
DOMAINES = {
    "Tech": 52000,
    "Banque": 58000,
    "Conseil": 55000,
    "Industrie": 45000,
    "Santé": 48000,
    "Marketing": 42000,
    "Juridique": 50000,
}

POSTES_PAR_NIVEAU = {
    1: ["Junior Developer", "Analyste Junior", "Assistant", "Consultant Junior"],
    2: ["Developer", "Analyste Confirmé", "Chargé de mission", "Consultant"],
    3: ["Senior Developer", "Analyste Senior", "Expert", "Consultant Senior"],
    4: ["Engineering Manager", "Manager", "Chef de projet", "Team Lead"],
    5: ["Directeur Technique", "Directeur", "Directeur de division"],
    6: ["CTO", "CEO", "CFO", "Directeur Général"],
}


def _niveau_vers_poste(niveau: int, rng: np.random.Generator) -> str:
    return rng.choice(POSTES_PAR_NIVEAU[niveau])


def generer_donnees(
    n: int = 1500,
    seed: int = 42,
    taux_valeurs_manquantes: float = 0.05,
) -> pd.DataFrame:
    """Génère un DataFrame synthétique de `n` diplômés.

    Introduit une corrélation entre le prestige de l'université et la
    réussite (salaire, niveau hiérarchique, vitesse de progression).
    """
    rng = np.random.default_rng(seed)

    univ_noms = list(UNIVERSITES.keys())
    univ_prestige = np.array([UNIVERSITES[u] for u in univ_noms])
    # Les universités prestigieuses ont davantage de diplômés échantillonnés.
    poids_univ = univ_prestige / univ_prestige.sum()

    domaine_noms = list(DOMAINES.keys())

    lignes = []
    for i in range(n):
        u_idx = rng.choice(len(univ_noms), p=poids_univ)
        universite = univ_noms[u_idx]
        prestige = univ_prestige[u_idx]

        domaine = rng.choice(domaine_noms)
        mediane_dom = DOMAINES[domaine]

        annee_diplome = int(rng.integers(2005, 2023))
        annees_experience = max(0, 2024 - annee_diplome - int(rng.integers(0, 3)))

        # --- Niveau hiérarchique ---------------------------------------
        # Base croissante avec l'expérience + bonus de prestige + bruit.
        niveau_continu = (
            1.0
            + 0.28 * annees_experience
            + 2.5 * prestige
            + rng.normal(0, 0.9)
        )
        niveau = int(np.clip(round(niveau_continu), 1, 6))

        # --- Salaire ----------------------------------------------------
        # Progression avec le niveau et l'expérience + effet prestige modéré.
        salaire = (
            mediane_dom
            * (1.0 + 0.16 * (niveau - 1))
            * (1.0 + 0.012 * annees_experience)
            * (0.90 + 0.28 * prestige)
            * rng.normal(1.0, 0.11)
        )
        salaire = int(round(max(28000, salaire) / 500) * 500)

        taille_entreprise = int(
            np.clip(rng.lognormal(mean=5.5, sigma=1.6), 5, 250000)
        )

        satisfaction = int(np.clip(round(rng.normal(6.5 + prestige, 1.8)), 1, 10))

        lignes.append(
            {
                "id": i + 1,
                "nom": f"Personne_{i + 1:05d}",
                "universite": universite,
                "annee_diplome": annee_diplome,
                "domaine_formation": domaine,
                "poste_actuel": _niveau_vers_poste(niveau, rng),
                "domaine_professionnel": domaine,
                "annees_experience": annees_experience,
                "salaire_annuel": salaire,
                "niveau_poste": niveau,
                "taille_entreprise": taille_entreprise,
                "satisfaction_carriere": satisfaction,
            }
        )

    df = pd.DataFrame(lignes)

    # --- Injection de valeurs manquantes / bruit de nommage ------------
    df = _injecter_imperfections(df, rng, taux_valeurs_manquantes)
    return df


def _injecter_imperfections(
    df: pd.DataFrame, rng: np.random.Generator, taux: float
) -> pd.DataFrame:
    """Rend le jeu de données réaliste : NaN, doublons, variantes de noms."""
    n = len(df)

    # Valeurs manquantes sur colonnes optionnelles.
    for col in ["satisfaction_carriere", "nom"]:
        mask = rng.random(n) < taux
        df.loc[mask, col] = np.nan

    # Quelques salaires manquants.
    mask = rng.random(n) < taux / 2
    df.loc[mask, "salaire_annuel"] = np.nan

    # Variantes de casse / abréviations pour tester la normalisation.
    variantes = {
        "Université Paris-Saclay": "UNIVERSITÉ PARIS-SACLAY",
        "École Polytechnique": "Ecole Polytechnique",
        "Sorbonne Université": "sorbonne universite",
    }
    for idx in df.index:
        u = df.at[idx, "universite"]
        if u in variantes and rng.random() < 0.3:
            df.at[idx, "universite"] = variantes[u]

    # Introduire quelques doublons francs.
    n_dups = max(1, int(n * 0.01))
    dups = df.sample(n=n_dups, random_state=int(rng.integers(0, 1_000_000)))
    df = pd.concat([df, dups], ignore_index=True)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Générateur de données synthétiques")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="data/input/donnees_diplomes.csv",
    )
    args = parser.parse_args()

    df = generer_donnees(n=args.n, seed=args.seed)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"{len(df)} lignes générées -> {args.output}")
