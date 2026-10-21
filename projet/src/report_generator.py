"""
Génération du rapport final (Markdown) et de l'export JSON des statistiques.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .statistical_analysis import interpreter_force


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _conclusion_globale(stats_: dict) -> str:
    """Synthèse : existe-t-il une corrélation significative ?"""
    signaux = []
    anova = stats_.get("anova_globale", {})
    if anova.get("applicable") and anova.get("significatif"):
        signaux.append("ANOVA globale significative")
    reg = stats_.get("regression", {})
    effet = reg.get("effet_universite", {}) if reg.get("applicable") else {}
    if effet.get("significatif"):
        signaux.append("effet université significatif en régression contrôlée")
    chi2 = stats_.get("chi2", {})
    if chi2.get("applicable") and chi2.get("significatif"):
        signaux.append("dépendance Chi-carré significative")

    if len(signaux) >= 2:
        return (
            "**OUI** — Il existe une corrélation statistiquement significative "
            "entre l'université de formation et la réussite professionnelle. "
            f"Signaux concordants : {', '.join(signaux)}."
        )
    if len(signaux) == 1:
        return (
            "**PARTIELLEMENT** — Un signal significatif détecté "
            f"({signaux[0]}), mais les preuves ne sont pas pleinement "
            "concordantes. À interpréter avec prudence."
        )
    return (
        "**NON** — Aucune corrélation significative robuste n'a été détectée "
        "entre l'université et la réussite professionnelle dans cet échantillon."
    )


def construire_rapport_markdown(
    df: pd.DataFrame,
    stats_: dict,
    rapport_nettoyage: dict,
    chemins_visus: list[str],
) -> str:
    lignes: list[str] = []
    A = lignes.append

    A("# Rapport d'analyse : Université et réussite professionnelle")
    A("")
    A(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*")
    A("")

    # --- Section 1 : Résumé exécutif ---------------------------------
    A("## Section 1 — Résumé exécutif")
    A("")
    A(f"- **Observations analysées** : {len(df)}")
    A(f"- **Universités couvertes** : {df['universite'].nunique()}")
    A(
        f"- **Domaines professionnels couverts** : "
        f"{df['domaine_professionnel'].nunique()} "
        f"({', '.join(sorted(df['domaine_professionnel'].unique()))})"
    )
    if rapport_nettoyage.get("universites_exclues"):
        A(
            f"- **Universités exclues** (< "
            f"{rapport_nettoyage.get('min_observations')} obs.) : "
            f"{len(rapport_nettoyage['universites_exclues'])}"
        )
    A("")
    A("### Conclusion principale")
    A("")
    A(_conclusion_globale(stats_))
    A("")

    # --- Section 2 : Classements par domaine -------------------------
    A("## Section 2 — Classements par domaine")
    A("")
    classement = stats_.get("classement_par_domaine", {})
    for domaine, lignes_c in classement.items():
        A(f"### DOMAINE : {domaine}")
        A("```")
        A("-" * 40)
        A("Top 5 des universités :")
        for i, l in enumerate(lignes_c[:5], start=1):
            ic = l["ic95_demi_largeur"]
            ic_txt = f"±{ic}" if ic is not None else "n/a"
            A(
                f"{i}. {l['universite']} - Score: {l['score_moyen']:.2f} "
                f"({ic_txt}), n={l['n']} diplômés"
            )
            A(f"   • Salaire médian : {l['salaire_median']/1000:.0f} k€")
            A(f"   • % dans top 25% : {l['taux_top25']:.0f}%")
            d = l["cohen_d_vs_reste"]
            d_txt = (
                f"{d} ({interpreter_force(d, 'd')})" if d is not None else "n/a"
            )
            A(
                f"   • Significativité : p={_fmt_p(l['p_value_vs_reste'])}"
                f" | Cohen's d={d_txt}"
            )
        A("-" * 40)
        A("```")
        A("")

    # --- Section 3 : Analyse statistique globale ---------------------
    A("## Section 3 — Analyse statistique globale")
    A("")

    anova = stats_.get("anova_globale", {})
    A("### ANOVA (score ~ université, toutes données)")
    if anova.get("applicable"):
        A(
            f"- F-statistic = {anova['f_statistic']:.3f} | "
            f"p-value = {_fmt_p(anova['p_value'])} "
            f"→ {'significatif' if anova['significatif'] else 'non significatif'}"
        )
        A(f"- Universités comparées : {anova['n_universites']}")
    else:
        A("- Non applicable (données insuffisantes).")
    A("")

    A("### ANOVA par domaine")
    A("")
    A("| Domaine | F | p-value | Significatif | n univ. |")
    A("|---|---|---|---|---|")
    for dom, r in stats_.get("anova_par_domaine", {}).items():
        if r.get("applicable"):
            A(
                f"| {dom} | {r['f_statistic']:.2f} | {_fmt_p(r['p_value'])} | "
                f"{'oui' if r['significatif'] else 'non'} | "
                f"{r['n_universites']} |"
            )
        else:
            A(f"| {dom} | — | — | n/a | {r.get('n_universites', 0)} |")
    A("")

    chi2 = stats_.get("chi2", {})
    A("### Test du Chi-carré (université × catégorie de réussite)")
    if chi2.get("applicable"):
        A(
            f"- χ² = {chi2['chi2']:.2f}, dl = {chi2['degres_liberte']}, "
            f"p-value = {_fmt_p(chi2['p_value'])}"
        )
        A(
            f"- V de Cramér = {chi2['cramer_v']:.3f} "
            f"(association {interpreter_force(chi2['cramer_v'], 'cramer')})"
        )
        A(
            f"- Conclusion : "
            f"{'dépendance significative' if chi2['significatif'] else 'indépendance non rejetée'}"
        )
    else:
        A(f"- Non applicable ({chi2.get('raison', 'n/a')}).")
    A("")

    sp = stats_.get("spearman", {})
    A("### Corrélation de Spearman (rang université vs score)")
    if sp.get("applicable"):
        A(
            f"- ρ = {sp['rho']:.3f}, p-value = {_fmt_p(sp['p_value'])} "
            f"(corrélation {interpreter_force(sp['rho'], 'rho')})"
        )
    A("")

    reg = stats_.get("regression", {})
    A("### Régression linéaire multiple")
    A("")
    A(
        "Modèle : `score_global ~ C(université) + années_expérience "
        "+ C(domaine) + log(taille_entreprise)`"
    )
    if reg.get("applicable"):
        A(f"- R² = {reg['r_squared']:.3f} | R² ajusté = {reg['r_squared_adj']:.3f}")
        A(
            f"- F global = {reg['f_statistic']:.2f}, "
            f"p-value = {_fmt_p(reg['f_pvalue'])}"
        )
        effet = reg.get("effet_universite", {})
        if "p_value" in effet:
            A(
                f"- **Effet net de l'université** (test F type II) : "
                f"F = {effet['F']:.2f}, p-value = {_fmt_p(effet['p_value'])} "
                f"→ {'significatif' if effet['significatif'] else 'non significatif'}"
            )
        A("")
        A(
            "> Interprétation : à expérience, domaine et taille d'entreprise "
            "égaux, l'université "
            + (
                "explique une part significative de la variation du score."
                if effet.get("significatif")
                else "n'explique pas de part significative supplémentaire."
            )
        )
    A("")

    # --- Section 4 : Insights et recommandations ---------------------
    A("## Section 4 — Insights et recommandations")
    A("")
    insights = _calculer_insights(df, stats_)

    A("### Universités spécialistes (excellentes dans 1-2 domaines)")
    for u, doms in insights["specialistes"]:
        A(f"- **{u}** : se démarque en {', '.join(doms)}")
    if not insights["specialistes"]:
        A("- Aucune université nettement spécialiste détectée.")
    A("")

    A("### Universités généralistes (bonnes partout)")
    for u, score in insights["generalistes"]:
        A(f"- **{u}** : score moyen {score:.1f} sur l'ensemble des domaines")
    if not insights["generalistes"]:
        A("- Aucune université nettement généraliste détectée.")
    A("")

    A("### Domaines où l'université fait la plus grande différence")
    for dom, f in insights["domaines_differenciants"]:
        A(f"- **{dom}** (F ANOVA = {f:.1f})")
    A("")

    A("### Domaines où l'université n'a pas d'impact significatif")
    if insights["domaines_non_significatifs"]:
        for dom in insights["domaines_non_significatifs"]:
            A(f"- **{dom}**")
    else:
        A("- Aucun : l'université a un effet significatif dans tous les domaines testés.")
    A("")

    # --- Annexe : visualisations -------------------------------------
    A("## Annexe — Visualisations générées")
    A("")
    import os

    for c in chemins_visus:
        nom = os.path.basename(c)
        chemin_rel = os.path.join("visualizations", nom)
        A(f"![{nom}]({chemin_rel})")
        A("")

    return "\n".join(lignes)


def _calculer_insights(df: pd.DataFrame, stats_: dict) -> dict:
    """Déduit spécialistes / généralistes / domaines différenciants."""
    pivot = df.pivot_table(
        index="universite",
        columns="domaine_professionnel",
        values="score_global",
        aggfunc="mean",
    )

    # Spécialistes : au moins un domaine très au-dessus de leur propre moyenne
    # ET au-dessus de la moyenne du domaine.
    moyenne_domaine = pivot.mean(axis=0)
    specialistes = []
    for u in pivot.index:
        ligne = pivot.loc[u]
        ecart = ligne - moyenne_domaine
        tops = ecart.dropna().sort_values(ascending=False)
        doms_forts = [d for d, v in tops.items() if v > 5][:2]
        moy_u = ligne.mean()
        # Spécialiste si fort sur 1-2 domaines mais pas globalement dominant.
        if doms_forts and moy_u < moyenne_domaine.mean() + 5:
            specialistes.append((u, doms_forts))

    # Généralistes : score moyen élevé sur l'ensemble (top 3).
    moyennes_glob = pivot.mean(axis=1).sort_values(ascending=False)
    seuil = moyennes_glob.mean()
    generalistes = [
        (u, s)
        for u, s in moyennes_glob.items()
        if s >= seuil
    ][:3]

    # Domaines différenciants : F ANOVA le plus élevé.
    anova_dom = stats_.get("anova_par_domaine", {})
    differenciants = sorted(
        [
            (d, r["f_statistic"])
            for d, r in anova_dom.items()
            if r.get("applicable")
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    non_signif = [
        d
        for d, r in anova_dom.items()
        if r.get("applicable") and not r.get("significatif")
    ]

    return {
        "specialistes": specialistes[:5],
        "generalistes": generalistes,
        "domaines_differenciants": differenciants,
        "domaines_non_significatifs": non_signif,
    }


# --------------------------------------------------------------------------
# Export JSON
# --------------------------------------------------------------------------
def _json_safe(obj):
    """Convertit récursivement les objets non sérialisables."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def exporter_json(stats_: dict, rapport_nettoyage: dict, chemin: str) -> None:
    # On retire le texte volumineux du summary de régression du JSON principal.
    stats_copie = json.loads(json.dumps(_json_safe(stats_)))
    if "regression" in stats_copie and isinstance(stats_copie["regression"], dict):
        stats_copie["regression"].pop("summary_text", None)

    payload = {
        "nettoyage": _json_safe(rapport_nettoyage),
        "statistiques": stats_copie,
    }
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
