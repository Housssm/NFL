#!/usr/bin/env python3
"""
Script principal — Analyse de corrélation Université / Réussite professionnelle.

Usage :
    python main.py --input data/input/donnees_diplomes.csv
    python main.py --generate            # génère d'abord des données synthétiques
    python main.py --input fichier.csv --min-obs 30

Produit :
    data/output/scores_calcules.csv   (données enrichies des scores)
    data/output/statistiques.json     (toutes les statistiques)
    visualizations/*.png              (graphiques)
    rapport_final.md                  (rapport Markdown)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

# Permet l'exécution directe (python main.py) comme en module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_processing, metrics, statistical_analysis, visualizations
from src import report_generator
from src.data_generator import generer_donnees

BASE = os.path.dirname(os.path.abspath(__file__))
DOSSIER_OUTPUT = os.path.join(BASE, "data", "output")
DOSSIER_VISUS = os.path.join(BASE, "visualizations")
DOSSIER_INPUT = os.path.join(BASE, "data", "input")


def _log(msg: str) -> None:
    print(f"[analyse] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse Université / Réussite professionnelle"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(DOSSIER_INPUT, "donnees_diplomes.csv"),
        help="Chemin du CSV d'entrée.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Génère un jeu de données synthétique avant l'analyse.",
    )
    parser.add_argument("--n", type=int, default=1500, help="Taille si --generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-obs",
        type=int,
        default=30,
        help="Nombre minimum d'observations par université (filtrage).",
    )
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(DOSSIER_OUTPUT, exist_ok=True)
    os.makedirs(DOSSIER_VISUS, exist_ok=True)
    os.makedirs(DOSSIER_INPUT, exist_ok=True)

    # --- Étape 0 : génération optionnelle -----------------------------
    if args.generate or not os.path.exists(args.input):
        if not args.generate and not os.path.exists(args.input):
            _log(
                f"Fichier introuvable ({args.input}). "
                "Génération d'un jeu synthétique."
            )
        _log(f"Génération de {args.n} observations synthétiques...")
        df_gen = generer_donnees(n=args.n, seed=args.seed)
        df_gen.to_csv(args.input, index=False, encoding="utf-8")
        _log(f"Données écrites dans {args.input}")

    # --- Étape 1 : chargement -----------------------------------------
    _log(f"Chargement de {args.input}")
    df = pd.read_csv(args.input)
    _log(f"{len(df)} lignes brutes chargées.")

    # --- Étape 2 : nettoyage ------------------------------------------
    _log("Nettoyage et préparation des données...")
    df, rapport_nettoyage = data_processing.nettoyer(
        df, min_observations=args.min_obs
    )
    _log(
        f"{rapport_nettoyage['lignes_finales']} lignes retenues, "
        f"{len(rapport_nettoyage['universites_retenues'])} universités."
    )
    if rapport_nettoyage["lignes_finales"] == 0:
        _log("ERREUR : aucune donnée après nettoyage/filtrage. Abandon.")
        return 1

    # --- Étape 3 : scores ---------------------------------------------
    _log("Calcul des scores de réussite...")
    df = metrics.calculer_scores(df)
    df = metrics.categoriser_reussite(df, n_categories=3)

    # Exports tabulaires.
    chemin_scores = os.path.join(DOSSIER_OUTPUT, "scores_calcules.csv")
    df.to_csv(chemin_scores, index=False, encoding="utf-8")
    resume = metrics.resume_par_universite_domaine(df)
    resume.to_csv(
        os.path.join(DOSSIER_OUTPUT, "resume_universite_domaine.csv"),
        index=False,
        encoding="utf-8",
    )
    _log(f"Scores exportés -> {chemin_scores}")

    # --- Étape 4 : statistiques ---------------------------------------
    _log("Exécution des tests statistiques...")
    stats_ = statistical_analysis.analyser(df)

    # --- Étape 5 : visualisations -------------------------------------
    _log("Génération des visualisations...")
    chemins_visus = visualizations.generer_toutes(
        df, stats_["classement_par_domaine"], DOSSIER_VISUS
    )
    _log(f"{len(chemins_visus)} graphiques générés.")

    # --- Étape 6 : rapport + JSON -------------------------------------
    _log("Rédaction du rapport...")
    md = report_generator.construire_rapport_markdown(
        df, stats_, rapport_nettoyage, chemins_visus
    )
    chemin_rapport = os.path.join(BASE, "rapport_final.md")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
        f.write(md)

    chemin_json = os.path.join(DOSSIER_OUTPUT, "statistiques.json")
    report_generator.exporter_json(stats_, rapport_nettoyage, chemin_json)

    dt = time.time() - t0
    _log(f"Rapport      -> {chemin_rapport}")
    _log(f"Statistiques -> {chemin_json}")
    _log(f"Terminé en {dt:.1f} s.")

    # Résumé console de la conclusion.
    print("\n" + "=" * 60)
    print("CONCLUSION :")
    print(report_generator._conclusion_globale(stats_))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
