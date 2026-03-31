from IPython.display import Markdown, display
import pandas as pd
import numpy as np
from numbers import Real

import unicodedata
import re
import inspect
from dataclasses import dataclass, field
from typing import Any, Optional, Set, Literal, Union, List, Dict, Tuple, Iterable, Sequence
import matplotlib.pyplot as plt
import seaborn as sns


_VERBOSE_ICONS = {
    "info": "ℹ️",
    "quartiles": "🍕",
    "bounds": "🚧",
    "outliers": "☢️",
    "examples": "🔍",
    "result": "✅",
    "update": "🔄️",
    "warning": "⚠️",
}


def _emit(verbose: bool, icon_key: str, message: str) -> None:
    """Emit a verbose message prefixed with an icon when enabled."""

    if not verbose:
        return

    icon = _VERBOSE_ICONS.get(icon_key, _VERBOSE_ICONS["info"])
    print(f"{icon} {message}")


def _normalise_quantiles(q: Iterable[float]) -> Tuple[float, float]:
    """Normalise the provided iterable of quantiles into a ``(low, high)`` tuple."""

    if isinstance(q, Sequence):
        quantiles = list(q)
    else:
        quantiles = list(q)  # type: ignore[arg-type]

    if len(quantiles) < 2:
        raise ValueError("q parameter should contain at least 2 elements")

    low, high = quantiles[0], quantiles[1]
    if not (0 <= low <= 1 and 0 <= high <= 1):
        raise ValueError("quantile values must be between 0 and 1")

    if low > high:
        raise ValueError("quantile lower bound must not exceed upper bound")

    return float(low), float(high)

def _ensure_label_list(value: Any) -> list[str]:
    """Coerce a stored label collection into a standalone list of strings."""

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, (tuple, set)):
        return [str(item) for item in value]

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    return [str(value)]

def add_malus_score(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    score: int = 1,
    *,
    label_col: str = "malus_labels",
    score_col: str = "malus_score",
    verbose: bool = False,
) -> pd.DataFrame:
    """Add a malus label and score to the rows selected by ``mask``.

    Parameters
    ----------
    df:
        DataFrame that will receive the malus updates. The object is modified in
        place and returned for convenience.
    mask:
        Boolean Series indicating which rows should receive the malus. The mask
        is aligned on ``df.index`` and any missing positions are treated as
        ``False``.
    label:
        Name of the malus to append to ``label_col`` when it is not already
        present for a row.
    score:
        Numeric value to add to ``score_col`` for each affected row. Defaults to
        ``1``.
    label_col:
        Name of the column that stores the collection of malus labels. When the
        column does not exist it is created with empty lists.
    score_col:
        Name of the numeric column accumulating the malus score. When missing it
        is initialised to ``0``.
    verbose:
        When ``True``, emit notebook-friendly diagnostics describing the
        operation.

    Returns
    -------
    pd.DataFrame
        The ``df`` reference, updated with the malus information.
    """

    if not isinstance(mask, pd.Series):
        raise TypeError("mask must be a pandas Series of booleans")

    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")

    if not isinstance(score, Real):
        raise TypeError("score must be a numeric value")

    if pd.isna(score):
        raise ValueError("score must not be NaN")

    mask_aligned = mask.reindex(df.index, fill_value=False)
    mask_values = mask_aligned.dropna()
    if not mask_values.isin([True, False]).all():
        raise ValueError("mask must contain only boolean values")

    mask_bool = mask_aligned.fillna(False).astype(bool)

    selected = int(mask_bool.sum())
    _emit(
        verbose,
        "info",
        (
            f"Preparing to apply malus '{label}' with score {score} "
            f"to {selected} row(s)."
        ),
    )

    if label_col not in df.columns:
        df[label_col] = [[] for _ in range(len(df))]
        _emit(verbose, "update", f"Initialised '{label_col}' column with empty labels.")
    else:
        df[label_col] = [_ensure_label_list(value) for value in df[label_col]]

    if score_col not in df.columns:
        df[score_col] = 0
        _emit(verbose, "update", f"Initialised '{score_col}' column with zeros.")
    else:
        df[score_col] = pd.to_numeric(df[score_col], errors="raise").fillna(0)

    if selected == 0:
        _emit(verbose, "result", "No rows matched the provided mask.")
        return df

    applied = 0
    skipped = 0
    for idx in df.index[mask_bool]:
        labels = df.at[idx, label_col]
        if label in labels:
            skipped += 1
            continue

        df.at[idx, label_col] = [*labels, label]
        df.at[idx, score_col] = df.at[idx, score_col] + score
        applied += 1

    _emit(
        verbose,
        "result",
        (
            f"Applied malus '{label}' to {applied} row(s). "
            f"Skipped {skipped} already containing the label."
        ),
    )

    return df

def compute_outlier_mask(
    series: pd.Series,
    compute_method: str | None = None,
    q: Iterable[float] | None = None,
    verbose: bool = False,
    iqr_coefficient: float = 1.5,
) -> pd.Series:
    """Return a boolean mask marking outliers according to the bounds.

    The function mirrors :func:`compute_outlier_count` by supporting both IQR and
    quantile-based bounds while returning a boolean Series aligned with
    ``series``.
    """

    if compute_method is None:
        compute_method = "quantile" if q is not None else "iqr"

    if q is not None and compute_method != "quantile":
        raise ValueError(
            "q parameter requires compute_method to be 'quantile' or None"
        )

    cleaned = series.dropna()
    dropped = len(series) - len(cleaned)
    _emit(verbose, "info", f"Removed {dropped} missing value(s) before analysis.")
    if cleaned.empty:
        _emit(verbose, "result", "No data available after dropping missing values.")
        return pd.Series(False, index=series.index, dtype=bool)

    lower, upper = get_outlier_bounds(
        cleaned,
        compute_method=compute_method,
        q=q,
        verbose=verbose,
        iqr_coefficient=iqr_coefficient,
    )

    mask = (series < lower) | (series > upper)
    mask_filled = mask.fillna(False)
    outliers = int(mask_filled.sum())
    pct_outliers = mask.mean(skipna=True) * 100 if len(series) else 0.0
    _emit(
        verbose,
        "result",
        (
            f"Generated outlier mask with {outliers} positive flag(s) "
            f"({pct_outliers:.1f}% of observations)."
        ),
    )

    return mask_filled.astype(bool)

def compute_outlier_count(
    series: pd.Series,
    compute_method: Literal["iqr", "quantile", None] = None,
    q: Iterable[float] | None = None,
    verbose: bool = False,
    iqr_coefficient: float = 1.5,
) -> int:
    """Count the number of outliers in ``series``.

    Parameters
    ----------
    series:
        Values to inspect for outliers. Missing values are ignored.
    compute_method:
        Strategy for computing the bounds. ``"iqr"`` uses the interquartile range
        and ``"quantile"`` expects explicit quantile bounds via ``q``. When left as
        ``None`` the method defaults to ``"quantile"`` if ``q`` is provided,
        otherwise ``"iqr"`` is used.
    q:
        Iterable containing the lower and upper quantile bounds used when
        ``compute_method`` is ``"quantile"``. At least two values must be provided.

    verbose:
        When ``True``, print diagnostic information with icons describing the
        computation steps.
    iqr_coefficient:
        Multiplier applied to the IQR when ``compute_method`` is ``"iqr"``.

    Returns
    -------
    int
        Number of values considered outliers.
    """

    if compute_method is None:
        compute_method = "quantile" if q is not None else "iqr"

    if q is not None and compute_method != "quantile":
        raise ValueError(
            "q parameter requires compute_method to be 'quantile' or None"
        )

    cleaned = series.dropna()
    dropped = len(series) - len(cleaned)
    _emit(verbose, "info", f"Removed {dropped} missing value(s) before analysis.")
    if cleaned.empty:
        _emit(verbose, "result", "No data available after dropping missing values.")
        return 0

    lower, upper = get_outlier_bounds(
        cleaned,
        compute_method=compute_method,
        q=q,
        verbose=verbose,
        iqr_coefficient=iqr_coefficient,
    )

    mask = (cleaned < lower) | (cleaned > upper)
    outliers = int(mask.sum())
    _emit(verbose, "result", f"Identified {outliers} outlier(s).")
    return outliers

def get_outlier_bounds(
    series: pd.Series,
    compute_method: Literal["iqr", "quantile", None] = None,
    q: Iterable[float] | None = None,
    verbose: bool = False,
    iqr_coefficient: float = 1.5
) -> Tuple[float, float]:
    """Return the lower and upper bounds for detecting outliers.

    Parameters
    ----------
    series:
        Series whose bounds are being calculated. Missing values are ignored.
    compute_method:
        Strategy for computing the bounds. ``"iqr"`` uses the interquartile range
        and ``"quantile"`` expects explicit quantile bounds via ``q``. When left as
        ``None`` the method defaults to ``"quantile"`` if ``q`` is provided,
        otherwise ``"iqr"`` is used.
    q:
        Iterable containing the lower and upper quantile bounds used when
        ``compute_method`` is ``"quantile"``. At least two values must be provided.
    verbose:
        When ``True``, print diagnostic information with icons describing the
        computation steps.
    iqr_coefficient:
        Multiplier applied to the IQR when ``compute_method`` is ``"iqr"``.

    Returns
    -------
    Tuple[float, float]
        ``(lower, upper)`` bounds beyond which values are considered outliers.
    """

    cleaned = series.dropna()
    if cleaned.empty:
        _emit(verbose, "result", "No data available after dropping missing values.")
        return float("nan"), float("nan")

    if compute_method is None:
        compute_method = "quantile" if q is not None else "iqr"

    if q is not None and compute_method != "quantile":
        raise ValueError(
            "q parameter requires compute_method to be 'quantile' or None"
        )

    if compute_method == "iqr":
        if iqr_coefficient <= 0:
            raise ValueError("iqr_coefficient must be positive")

        q1, q3 = cleaned.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - iqr_coefficient * iqr
        upper = q3 + iqr_coefficient * iqr

        _emit(verbose, "quartiles", f"Q1={q1} | Q3={q3} | IQR={iqr}")
        _emit(
            verbose,
            "bounds",
            (
                "Using IQR bounds: "
                f"coef={iqr_coefficient}, lower={lower}, upper={upper}"
            ),
        )
    elif compute_method == "quantile":
        if q is None:
            raise ValueError("q parameter is required when compute_method='quantile'")

        lower_quantile, upper_quantile = _normalise_quantiles(q)
        quantile_bounds = cleaned.quantile([lower_quantile, upper_quantile])
        lower, upper = tuple(quantile_bounds.tolist())

        _emit(
            verbose,
            "bounds",
            (
                "Using quantile bounds: "
                f"q_low={lower_quantile} -> {lower}, "
                f"q_high={upper_quantile} -> {upper}"
            ),
        )
    else:
        raise ValueError("Unsupported compute_method: {0}".format(compute_method))

    mask = (cleaned < lower) | (cleaned > upper)
    outliers = int(mask.sum())
    pct_outliers = (outliers / len(cleaned)) * 100 if len(cleaned) else 0.0
    _emit(
        verbose,
        "outliers",
        f"Outliers: {outliers}/{len(cleaned)} ({pct_outliers:.1f}%)",
    )

    if verbose and outliers > 0:
        examples = cleaned[mask].head(5).tolist()

        def _format_example(value: object) -> str:
            if isinstance(value, Real):
                return f"{float(value):.1f}"
            return str(value)

        formatted = ", ".join(_format_example(value) for value in examples)
        remaining = outliers - len(examples)
        suffix = f" (+ {remaining} autres)" if remaining > 0 else ""
        _emit(verbose, "examples", f"Exemples: [{formatted}]{suffix}")

    return lower, upper


def get_highly_correlated_features(
    data: pd.DataFrame,
    method: Literal["pearson", "spearman"] = "pearson",
    threshold: float = 0.7,
    return_pairs: bool = False,
    verbose: bool = False,
    round_digits: int | None = None,
    return_df: bool = False 
)-> Union[list[str], list[tuple], pd.DataFrame]:
    """
    Identifie les colonnes fortement corrélées dépassant un certain seuil.

    Args:
        data (pd.DataFrame): DataFrame contenant les données numériques.
        method (str): Méthode de corrélation ('pearson' ou 'spearman').
        threshold (float): Seuil de corrélation absolue pour considérer une forte corrélation.
        return_pairs (bool): Si True, retourne les paires corrélées.
        verbose (bool): Si True, affiche un résumé clair et trié.
        return_df: Si True, retourne un DataFrame des paires. Si False, retourne la liste des colonnes à supprimer.

    Returns:
        list: Liste des colonnes à supprimer ou liste de paires corrélées (col1, col2, corr_value).
    """
    corr_matrix = data.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)
    
    # --- Mode paires corrélées ---
    if return_pairs:
        pairs = (
            upper_triangle.stack()  # enlève NaN et aplatit
            .reset_index()
            .rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: "Correlation"})
        )
        pairs = pairs.loc[pairs["Correlation"].abs() > threshold]
        pairs = pairs.sort_values("Correlation", key=lambda x: x.abs(), ascending=False)

        # Arrondir la colonne Correlation si round_digits n'est pas None
        if round_digits is not None:
            pairs["Correlation"] = pairs["Correlation"].round(round_digits)
        
        if verbose:
            n_pairs = len(pairs)
            if n_pairs == 0:
                print(f"✅ Aucune paire de colonnes corrélée au-delà de {threshold}")
            else:
                print(f"⚠️ {n_pairs} paires dépassent le seuil de {threshold} ({method}) :\n")
                display(pairs.reset_index(drop=True).head().style.hide(axis="index"))  # top 10 plus lisibles
        if return_df:
            return pairs.reset_index(drop=True)

        return list(pairs.itertuples(index=False, name=None))
    
    # --- Mode colonnes à supprimer ---
    hc_indicators = [col for col in upper_triangle.columns if (upper_triangle[col].abs() > threshold).any()]
    
    if verbose:
        if hc_indicators:
            print(f"⚠️ {len(hc_indicators)} colonnes corrélées dépassent {threshold} ({method}) :")
            print(", ".join(hc_indicators))
        else:
            print(f"✅ Aucune corrélation forte détectée (>{threshold})")
    
    return hc_indicators

def plot_correlation_triangle(data, 
                              method="pearson", 
                              title = "Corrélation matrix sans redondance", 
                              figsize=(12,10), ax=None, threshold = None):
    """
    Trace une matrice de corrélation triangulaire (heatmap) pour visualiser les relations.
    
    Args:
        data (pd.DataFrame): Données numériques à analyser.
        method (str): Méthode de corrélation ('pearson' ou 'spearman').
        title (str): Titre du graphique.
        figsize (tuple): Taille de la figure.
        ax (matplotlib.axes.Axes): Axe sur lequel tracer le graphique.
        threshold (float): Seuil pour masquer les corrélations faibles.
    """    
    
    corr_matrix = data.corr(method=method)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if threshold is not None:
        mask_threshold = corr_matrix.round(2).abs() <= threshold
        corr_matrix = corr_matrix.mask(mask_threshold)
        
    
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad(color="lightgray")
    
    plt.figure(figsize=figsize)        
    sns.heatmap(corr_matrix, ax=ax, 
            annot = True, fmt=".2f",
            cmap = cmap, 
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k = 1),
            square = True,
            center = 0, vmin = -1, vmax = 1,
            linewidths=0.5, linecolor='white',
            cbar = not(threshold) # not needed in threshold mode
    )

    ax.set_title(title)
    ax.tick_params(axis = "x", rotation = 90)

    if ax is None:
        plt.show()


def get_enhanced_info(df, sort_by="Fill Rate (%)", ascending=False, verbose=False):
    """
    Génère un résumé détaillé des colonnes d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        sort_by (str): Colonne pour trier le résumé.
        ascending (bool): Ordre du tri.
        verbose (bool): Si True, affiche des statistiques supplémentaires.

    Returns:
        pd.DataFrame: Un DataFrame contenant le résumé des colonnes.
    """
    if df.empty:
        return pd.DataFrame(columns=["Column", "Non-Null", "Fill Rate (%)", "Unique Count", "Dtype"])
    
    if verbose:
        print(f"Analyzing {len(df)} rows across {len(df.columns)} columns")
    
    # Calculate metrics
    non_null = df.notna().sum().rename("Non-Null")
    fill_rate = df.notna().mean().mul(100).round(2).rename("Fill Rate (%)")
    unique_counts = df.nunique().rename("Unique Count")
    dtypes = df.dtypes.rename("Dtype")
    
    # Combine into summary
    summary_df = (pd.concat([non_null, fill_rate, unique_counts, dtypes], axis=1)
                    .rename_axis("Column")
                    .reset_index())
    
    # Sort if requested
    if sort_by and sort_by in summary_df.columns:
        summary_df = summary_df.sort_values(by=sort_by, ascending=ascending)
    elif sort_by and sort_by not in summary_df.columns:
        print(f"Warning: Column '{sort_by}' not found. Returning unsorted.")
    
    if verbose:
        missing_cols = (summary_df['Fill Rate (%)'] < 100).sum()
        print(f"Found {missing_cols} columns with missing values")
    
    return summary_df[["Column", "Non-Null", "Fill Rate (%)", "Unique Count", "Dtype"]]

def to_padded_str(val, zero_pad=2, regex_to_remove=r'\.0$'):
    """
    Nettoie un code : enlève une partie avec regex, applique zfill,
    conserve NaN/None.
    - zero_pad : longueur souhaitée après padding
    - regex_to_remove : motif regex à retirer
    """
    if pd.isna(val) or val in ["None","none", "nan", "NaN", ""]:
        return np.nan
    
    return re.sub(regex_to_remove, '', str(val)).zfill(zero_pad)
 

def verify_column_reproducibility(df, target_col, compute_func, return_df=False):
    """
    Example of use: 
    _=verify_column_reproducibility(
        df_geo,
        "reg_id_old",
        lambda df: "R" + df["reg_code_old"].astype(str).str.zfill(2)
    )
    """
    check_col = target_col + "_check"
    temp_incoherences = None
    temp_df = df.assign(
        **{check_col: compute_func(df)}
    )
    df_compare = temp_df[target_col] == temp_df[check_col]
    n_total = len(df)
    n_errors = (~df_compare).sum()
    error_pct = n_errors / n_total * 100

    if df_compare.all():
        display(Markdown(f"✅ **{target_col}** peut être calculé à 100%"))
    else:
        display(Markdown(
            f"❌ **{target_col}** ne peut PAS être complètement calculé à 100%<br>"
            f"🟠 **{n_errors} erreurs ({error_pct:.1f}%)** sur {n_total} lignes"
        ))
        temp_incoherences = temp_df.loc[~df_compare,]
        if not return_df:
            display(temp_incoherences[[target_col, check_col]].head())        
    
    return temp_incoherences if return_df else None
       
def get_possible_candidates(df, verbose=False):
    candidates = []
    for col in df.columns:
        if df[col].nunique() == df.shape[0]:
            candidates.append(col)
    
    if verbose:
        display(Markdown("#### 🔍 Analyse des colonnes pour trouver des clés candidates"))
        
        if candidates:
            result = f"> **🎯 {len(candidates)} clé candidate(s) trouvée(s) :**\n>\n"
            for candidate in candidates:
                result += f"> - ✅ `{candidate}`\n"
            display(Markdown(result))
        else:
            display(Markdown("> **❌ Aucune clé candidate trouvée**"))
    
    return candidates if candidates else None

def format_french_underscore(x):
    return f"{x:_.2f}".replace('.', ',')

def display_stats(df, columns=None):
    # Gestion du paramètre columns
    if columns is None:
        # Toutes les colonnes
        cols_to_process = df.columns
    elif isinstance(columns, str):
        # Une seule colonne en string
        cols_to_process = [columns]
    elif isinstance(columns, list):
        # Liste de colonnes
        cols_to_process = columns
    else:
        raise ValueError("Le paramètre 'columns' doit être None, str ou list")

    # Vérifier que les colonnes existent
    missing_cols = [col for col in cols_to_process if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes inexistantes: {missing_cols}")

    for col in cols_to_process:
        display(Markdown("---"))
        display(Markdown(f"**Colonne: {col}**"))

        # Afficher le VRAI type
        print(f"🔍 Type réel: {df[col].dtype}")

        # Pour gérer les types datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"Période: {df[col].min()} à {df[col].max()}")

        # Pour détecter les "faux float" (entiers stockés en float)
        if df[col].dtype == 'float64':
            if ( (df[col] % 1 == 0) | (df[col].isna()) ).all():
                print("⚠️ Cette colonne pourrait être convertie en int")

        # Mémoire utilisée (utile en data engineering)
        memory_mb = df[col].memory_usage(deep=True) / 1024**2
        print(f"📊 Mémoire: {memory_mb:.2f} MB")

        # Statistiques adaptées selon le type
        if df[col].dtype in ['int64', 'float64']:
            describe_stats = df[[col]].describe()            
            display(describe_stats.style.format(format_french_underscore))
        else:
            # Pour les colonnes Object, afficher des stats pertinentes
            max_len = df[col].dropna().astype(str).str.len().max() # la longueur max du texte SANS tenir compte des NaN
            min_len = df[col].dropna().astype(str).str.len().min() # la longueur min du texte SANS tenir compte des NaN
            print(f"Longueur min du texte: {min_len}")
            print(f"Longueur max du texte: {max_len}")
            print(f"Valeurs uniques: {df[col].nunique()}")
            print(f"Valeurs les plus fréquentes:")
            display(pd.DataFrame(df[col].value_counts()).head())

        # Valeurs manquantes
        missing_count = df[col].isnull().sum()
        missing_percentage = df[col].isnull().mean().round(3)*100
        if missing_count > 0:
            print(f"⚠️ Il y a {missing_count} valeurs manquantes dans la colonne '{col}' soit {missing_percentage}%")
        else:
            print(f"✅ Pas de valeurs manquantes dans la colonne '{col}'")


def save_stats_to_csv(df, output_path="stats.csv", columns=None):
    """
    Sauvegarde les statistiques des colonnes d'un DataFrame dans un fichier CSV.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        output_path (str): Chemin du fichier CSV de sortie.
        columns (list|str|None): Colonnes à analyser (None = toutes).
    """
    # Gestion du paramètre columns
    if columns is None:
        cols_to_process = df.columns
    elif isinstance(columns, str):
        cols_to_process = [columns]
    elif isinstance(columns, list):
        cols_to_process = columns
    else:
        raise ValueError("Le paramètre 'columns' doit être None, str ou list")

    # Vérifier que les colonnes existent
    missing_cols = [col for col in cols_to_process if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes inexistantes: {missing_cols}")

    # Liste pour stocker les stats
    stats_list = []

    for col in cols_to_process:
        col_stats = {"colonne": col, "dtype": str(df[col].dtype)}

        # Type datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_stats["min"] = df[col].min()
            col_stats["max"] = df[col].max()

        # Faux float = entiers déguisés
        if df[col].dtype == 'float64':
            if ((df[col] % 1 == 0) | (df[col].isna())).all():
                col_stats["note"] = "⚠️ Colonne convertible en int"

        # Mémoire utilisée
        col_stats["memoire_MB"] = round(df[col].memory_usage(deep=True) / 1024**2, 3)

        # Statistiques numériques
        if df[col].dtype in ['int64', 'float64']:
            desc = df[col].describe()
            for stat_name, stat_value in desc.items():
                col_stats[stat_name] = stat_value
        else:
            # Colonnes catégoriques/objets
            col_stats["longueur_max"] = df[col].dropna().astype(str).str.len().max()
            col_stats["valeurs_uniques"] = df[col].nunique()
            top_values = df[col].value_counts().head(3).to_dict()
            col_stats["top_values"] = str(top_values)

        # Valeurs manquantes
        missing_count = df[col].isnull().sum()
        missing_percentage = df[col].isnull().mean() * 100
        col_stats["nb_nan"] = missing_count
        col_stats["pct_nan"] = round(missing_percentage, 2)

        stats_list.append(col_stats)

    # Conversion en DataFrame
    stats_df = pd.DataFrame(stats_list)

    # Sauvegarde en CSV
    stats_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Statistiques sauvegardées dans {output_path}")


def report_shape_changes(shape_before, shape_after):
    """
    Affiche un rapport sur les changements de dimensions (lignes/colonnes) entre deux formes.
    
    Args:
        shape_before (tuple): Forme initiale du DataFrame (ex: df.shape).
        shape_after (tuple): Forme finale du DataFrame.
    """
    
    print(f"Shape avant: {shape_before}")
    print(f"Shape après: {shape_after}")
    
    rows_diff = shape_before[0] - shape_after[0]
    cols_diff = shape_before[1] - shape_after[1]

    # Messages pour les lignes
    if rows_diff > 0:
        print(f"  ✂️  Lignes supprimées: {rows_diff}")
    elif rows_diff < 0:
        print(f"  ➕ Lignes ajoutées: {abs(rows_diff)}")
    
    
    if cols_diff > 0:
        print(f"  🗑️  Colonnes supprimées: {cols_diff}")
    elif cols_diff < 0:
        print(f"  📊 Colonnes ajoutées: {abs(cols_diff)}")
    
    if rows_diff == 0 and cols_diff == 0:
        print(f"  🔄 Aucun changement de dimension")

def normalize_commune(commune_name):
    """
    Normalise un nom de commune de manière robuste.
    1. Corrige les corruptions d'encodage spécifiques.
    2. Tente une réparation d'encodage générique.
    3. Supprime les articles définis en début de nom (Le, La, Les, L').
    4. Supprime les accents, gère les ligatures et standardise le format.
    """
    if pd.isna(commune_name):
        return ""

    text = str(commune_name).strip()

    # ÉTAPE 1 : CORRECTIONS MANUELLES POUR LES CAS SPÉCIFIQUES
    if "SchÄ°lcher" in text:
        text = text.replace("SchÄ°lcher", "Schoelcher")
    if "SchÅ“lcher" in text:
        text = text.replace("SchÅ“lcher", "Schoelcher")

    # ÉTAPE 2 : TENTATIVE DE RÉPARATION D'ENCODAGE GÉNÉRIQUE
    try:
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # ÉTAPE 3 : SUPPRESSION DES ARTICLES EN DÉBUT DE NOM (NOUVELLE ÉTAPE !)
    # Regex pour trouver "le ", "la ", "les ", "l'" au début, insensible à la casse
    pattern = r"^(le\s+|la\s+|les\s+|l')"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # ÉTAPE 4 : NORMALISATION STANDARD
    text = text.replace('œ', 'oe').replace('Œ', 'OE')
    nfkd_form = unicodedata.normalize('NFD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = re.sub(r'[\s-]+', ' ', text) # Remplace tirets et espaces multiples par un seul espace
    
    return text.upper().strip()

def print_md(*args):
    for item in args:
        if isinstance(item, (pd.DataFrame, pd.Series)):
            display(item)
        else:
            display(Markdown(str(item)))            

def get_duplicates_in_subset(df, columns_to_check = None, sort_results = True, verbose = False):
    """
    Trouve et retourne les lignes dupliquées basées sur un sous-ensemble de colonnes.

    Args:
        df (pd.DataFrame): Le DataFrame à vérifier.
        columns_to_check (list): Liste des colonnes définissant un duplicata.
        sort_results (bool): Si True, trie les résultats.
        verbose (bool): Si True, affiche le nombre de duplicatas trouvés.

    Returns:
        pd.DataFrame or None: Le DataFrame des duplicatas ou None si aucun n'est trouvé.
    """
    # Validation des inputs
    if columns_to_check:
        missing_cols = set(columns_to_check) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonnes inexistantes: {missing_cols}")
    else:
        columns_to_check = df.columns.to_list()

    duplicates_mask = df.duplicated(subset=columns_to_check, keep=False)
    duplicate_rows = df[duplicates_mask].copy()
    
    if verbose:
        if duplicate_rows.empty:
            print("✅ Aucun doublon trouvé")
        else:
            try:
                n_groups = duplicate_rows.groupby(columns_to_check).ngroups
                print(f"🔍 {len(duplicate_rows)} lignes dupliquées ({n_groups} groupes)")
            except TypeError:
                print(f"🔍 {len(duplicate_rows)} lignes dupliquées (groupes non calculables)")
    
    return duplicate_rows.sort_values(by=columns_to_check) if sort_results and not duplicate_rows.empty else duplicate_rows

def plot_regressions(df, x_cols, y_col, palette="Set2", 
                     ncols=None, figsize=None):
    """
    Plot multiple regression plots with auto colors AND layout
    
    Parameters:
    -----------
    df : pd.DataFrame
    x_cols : list
        Column names for X axes
    y_col : str
        Column name for Y axis
    palette : str
        Seaborn palette name
    ncols : int, optional
        Number of columns (default: auto based on n_plots)
    figsize : tuple, optional
        Figure size (default: auto based on layout)
    """
    n_plots = len(x_cols)
    colors = sns.color_palette(palette, n_plots)
    
    # 🎯 AUTO-CALCUL du layout optimal
    if ncols is None:
        ncols = min(3, n_plots)  # Max 3 colonnes par défaut
    
    nrows = int(np.ceil(n_plots / ncols))
    
    # 🎯 AUTO-CALCUL de la taille
    if figsize is None:
        width = ncols * 5  # 5 inches par colonne
        height = nrows * 4  # 4 inches par ligne
        figsize = (width, height)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes pour itération simple
    axes = np.array(axes).flatten()
    
    # Plot chaque régression
    for i, col in enumerate(x_cols):
        sns.regplot(
            data=df,
            x=col,
            y=y_col,
            ax=axes[i],
            color=colors[i]
        )
        axes[i].set_title(f"{y_col} vs {col}")
    
    # 🎯 Cacher les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_histograms(df, cols=None, palette="Set2", ncols=None, 
                    figsize=None, layout="auto", kde=True, bins="auto"):
    """
    Plot multiple histograms with auto colors and layout
    
    Parameters:
    -----------
    df : pd.DataFrame
    cols : list, optional
        Column names to plot (default: all numeric columns)
    palette : str
        Seaborn palette name
    ncols : int, optional
        Number of columns (default: auto based on n_plots)
    figsize : tuple, optional
        Figure size (default: auto based on layout)
    layout : str
        - "auto": smart grid (default)
        - "horizontal": force 1 row
        - "vertical": force 1 column
        - "square": try to make square grid
    kde : bool
        Show KDE curve (default: True)
    bins : int or str
        Number of bins (default: "auto")
    """
    # 🎯 AUTO-SELECT numeric columns if not specified
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_plots = len(cols)
    colors = sns.color_palette(palette, n_plots)
    
    # 🎯 SMART LAYOUT
    if ncols is None:
        if layout == "horizontal":
            ncols = n_plots
        elif layout == "vertical":
            ncols = 1
        elif layout == "square":
            ncols = int(np.ceil(np.sqrt(n_plots)))
        else:  # "auto"
            if n_plots <= 3:
                ncols = n_plots
            elif n_plots <= 6:
                ncols = 3
            else:
                ncols = 4
    
    nrows = int(np.ceil(n_plots / ncols))
    
    # 🎯 SMART FIGSIZE
    if figsize is None:
        width = ncols * 5
        height = nrows * 4
        figsize = (width, height)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()
    
    # 📊 Plot each histogram
    for i, col in enumerate(cols):
        sns.histplot(
            data=df,
            x=col,
            ax=axes[i],
            color=colors[i],
            kde=kde,
            bins=bins,
            alpha=0.7
        )
        axes[i].set_title(f"Distribution de {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Fréquence")
    
    # 🎯 Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def missing_summary(df: pd.DataFrame, numeric_only: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Résume les valeurs manquantes d'un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à analyser.
    numeric_only : bool, optional
        Si True, ne considère que les colonnes numériques.
    verbose : bool, optional
        Si True, affiche le résumé avant de le retourner.
    
    Returns
    -------
    pd.DataFrame
        Un DataFrame contenant :
        - col : nom de la colonne
        - missing_count : nombre de valeurs manquantes
        - missing_pct : pourcentage de valeurs manquantes
    """
    if numeric_only:
        df = df.select_dtypes(include="number")
    
    total_rows = len(df)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / total_rows * 100).round(2)

    summary = (
        pd.DataFrame({
            "col": missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values
        })
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(summary)
    
    return summary

def zero_negative_summary(df: pd.DataFrame, cols: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """
    Résume les valeurs nulles (zéros) et négatives d'un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à analyser.
    cols : list[str] | None, optional
        Liste des colonnes à vérifier. Si None, toutes les colonnes numériques sont utilisées.
    verbose : bool, optional
        Si True, affiche des messages via _emit().

    Returns
    -------
    pd.DataFrame
        Un DataFrame contenant :
        - col : nom de la colonne
        - zero_count : nombre de zéros
        - negative_count : nombre de valeurs négatives
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
        _emit(verbose, "info", f"{len(cols)} colonnes numériques sélectionnées automatiquement")

    zero_counts = (df[cols] == 0).sum()
    neg_counts = (df[cols] < 0).sum()

    summary = (
        pd.DataFrame({
            "col": cols,
            "zero_count": zero_counts.values,
            "negative_count": neg_counts.values
        })
        .query("zero_count > 0 | negative_count > 0")
        .reset_index(drop=True)
    )

    _emit(verbose, "result", f"{len(summary)} colonnes contiennent des valeurs nulles ou négatives")
    return summary



def check_formula_consistency(
    df: pd.DataFrame,
    formulas: Dict[str, str],
    atol: float = 0.1,
    rtol: float = 0.0,
    show_cols: Optional[List[str]] = None,
    verbose: bool = True,
    return_df: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Vérifie la cohérence entre les colonnes d'un DataFrame et un ensemble de formules.
    Example d'usage: 
    ----------
    formulas = {
        "GFATotal = GFABuildings + GFAParking": "`PropertyGFATotal` = `PropertyGFABuilding(s)` + `PropertyGFAParking`",
        "SiteEnergyUse(kBtu)": "`SiteEnergyUse(kBtu)` = (`Electricity(kWh)` * 3.412) + (`NaturalGas(therms)` * 100) + `SteamUse(kBtu)`",
    }

    df_results, df_inconsistencies = check_formula_consistency(
                        df_non_residential,
                        formulas=formulas,
                        show_cols=["PropertyName", "PrimaryPropertyType"],
                        return_df=True
    )

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame contenant les données à analyser.
    formulas : dict
        Dictionnaire {label: "target = expression"}.
    atol, rtol : float
        Tolérances absolue et relative pour np.isclose.
    show_cols : list[str] | None
        Colonnes additionnelles à afficher dans le DataFrame des incohérences.
    verbose : bool
        Si True, affiche un résumé détaillé et les avertissements.
    return_df : bool
        Si True, retourne deux DataFrames (résumé + incohérences).
    """

    if not isinstance(formulas, dict):
        raise TypeError("Le paramètre 'formulas' doit être un dictionnaire {label: 'target = expression'}.")

    results = []
    inconsistencies = []

    # Colonnes additionnelles à afficher
    show_cols = [col for col in (show_cols or []) if col in df.columns]

    # === BOUCLE PRINCIPALE ===
    for label, equation in formulas.items():

        # --- Validation de la syntaxe ---
        if "=" not in equation:
            raise ValueError(f"Formule invalide pour {label} : utilisez 'target = expression'.")

        target, expr = equation.split("=", 1)
        target = target.strip().strip("`")
        expr = expr.strip()

        if target not in df.columns:
            raise KeyError(f"Colonne cible '{target}' absente du DataFrame (formule {label}).")

        # --- Évaluation de la formule ---
        try:
            computed = df.eval(expr)
        except Exception as e:
            raise KeyError(f"Erreur lors de l’évaluation de la formule '{label}': {e}")

        # --- Masque pour exclure NaN / inf ---
        mask_valid = (
            df[target].notna()
            & computed.notna()
            & np.isfinite(df[target])
            & np.isfinite(computed)
        )

        # --- Comparaison ---
        match = np.isclose(df[target], computed, atol=atol, rtol=rtol)
        match &= mask_valid

        # --- Différences ---
        diff = (df[target] - computed).abs()
        mean_diff = diff[mask_valid].mean()
        mean_rel_diff = (diff[mask_valid] / computed[mask_valid].replace(0, np.nan)).mean()
        max_abs_diff = diff[mask_valid].max()
        consistency = match[mask_valid].mean() * 100 if mask_valid.any() else np.nan
        n_valid = mask_valid.sum()
        n_fail = (~match & mask_valid).sum()

        results.append({
            "formula_name": label,
            "target": target,
            "formula": expr,
            "max_abs_diff": max_abs_diff,
            "mean_diff": mean_diff,
            "mean_rel_diff_%": mean_rel_diff * 100,
            "consistency_%": consistency,
        })

        # --- Logs hiérarchiques ---
        if verbose:
            icon = "✅" if consistency == 100 else ("⚠️" if consistency >= 50 else "❌")
            print(f"\n{icon} [{label}]")
            print(f"│   ├─ Cible : {target}")
            print(f"│   ├─ Cohérence : {consistency:6.2f}%\t| Diff. relative : {mean_rel_diff * 100:6.2f}%")
            print(f"│   ├─ Diff. moyenne : {mean_diff:8.4f}\t| Max diff : {max_abs_diff:10.4f}")
            print(f"│   └─ Lignes valides : {n_valid:5d}\t| Incohérentes : {n_fail:5d}")

        # --- Extraction des incohérences ---
        mask_inconsistent = mask_valid & ~match
        if mask_inconsistent.any():
            cols_in_formula = [c for c in re.findall(r"`([^`]+)`", expr) if c in df.columns]

            # Colonnes contextuelles = celles des formules + celles demandées
            context_cols = list(dict.fromkeys(show_cols + [target] + cols_in_formula))
            context_cols = [c for c in context_cols if c in df.columns]

            # Sous-DataFrame des incohérences
            df_bad = df.loc[mask_inconsistent, context_cols].copy()
            df_bad["formula_name"] = label
            df_bad["row_index"] = df_bad.index
            df_bad["actual"] = df_bad[target]
            df_bad["expected"] = computed.loc[mask_inconsistent].to_numpy()
            df_bad["difference"] = (df_bad["actual"] - df_bad["expected"]).abs().round(4)
            df_bad["target"] = target
            df_bad["formula_fields"] = [cols_in_formula] * len(df_bad)            
            # df_bad["formula_fields"] = ", ".join(cols_in_formula)

            # Réorganisation lisible du DataFrame des incohérences
            base_cols = [
                "row_index", "formula_name", "target",
                "actual", "expected", "difference", "formula_fields"
            ]

            # Colonnes contextuelles (venues de show_cols)
            context_cols = [c for c in show_cols if c in df_bad.columns]

            # Colonnes issues de la formule
            formula_cols = [c for c in cols_in_formula if c in df_bad.columns]

            # Autres colonnes (par ex. non utilisées dans la formule)
            other_cols = [
                c for c in df_bad.columns
                if c not in base_cols + context_cols + formula_cols
            ]

            # Ordre final des colonnnes
            cols_order = base_cols + context_cols + formula_cols + other_cols
            df_bad = df_bad.loc[:, df_bad.columns.intersection(cols_order)]            

            num_cols = df_bad.select_dtypes(include=[np.number]).columns
            df_bad[num_cols] = df_bad[num_cols].round(4)

            inconsistencies.append(df_bad)

    # --- Résumé final ---
    df_results = pd.DataFrame(results)[[
        "formula_name", "target", "formula",
        "max_abs_diff", "mean_diff", "mean_rel_diff_%", "consistency_%"
    ]]

    df_inconsistencies = (
        pd.concat(inconsistencies, ignore_index=True)
        if inconsistencies
        else pd.DataFrame(columns = ["formula_name", "target", "actual", "expected", "difference"] + show_cols)
    )

    if return_df:
        return df_results, df_inconsistencies
    return df_results.set_index("formula_name")["consistency_%"].to_dict()


def get_existing_columns(df, columns_list, verbose=False):
    existing = [col for col in columns_list if col in df.columns]
    missing = [col for col in columns_list if col not in df.columns]
    
    if verbose and missing:
        print(f"⚠️ {len(missing)} colonne(s) manquante(s): {missing}")
    
    return existing

@dataclass 
class DataFrameGuard:
    """
        df_name : str -> nom de la variable DataFrame dans le scope appelant
        expect_rows / expect_cols : 'decrease', 'increase', 'same', or None
        verbose : bool -> affiche les colonnes ajoutées/supprimées même si tout est ok
    """
    df_name: str
    expect_rows: Optional[Literal['decrease', 'increase', 'same']] = None
    expect_cols: Optional[Literal['decrease', 'increase', 'same']] = None
    verbose: bool = False
    initial_shape: Optional[tuple] = field(default=None, init=False)
    initial_cols: Optional[Set[str]] = field(default=None, init=False)

    def __enter__(self):
        frame = inspect.currentframe().f_back
        df = frame.f_locals.get(self.df_name)

        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError(f"No DataFrame named '{self.df_name}' found in current scope.")

        self.initial_shape = df.shape
        self.initial_cols = set(df.columns)
        if self.initial_shape is None or self.initial_cols is None:
            raise RuntimeError(
                f"❌ DataFrameGuard initialization failed for '{self.df_name}'. "
                f"Cannot proceed with transformation for safety reasons."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False  # Laisse passer l'erreur d'origine
        # Vérification que le guard est correctement initialisé
        if self.initial_shape is None or self.initial_cols is None:
            raise RuntimeError(
                f"❌ DataFrameGuard was not properly initialized. "
                f"Transformation on '{self.df_name}' is BLOCKED for safety."
            )


        frame = inspect.currentframe().f_back
        df_after = frame.f_locals.get(self.df_name)

        # Vérification que le DataFrame existe toujours
        if df_after is None:
            raise RuntimeError(
                f"❌ DataFrame '{self.df_name}' disappeared during transformation! "
                f"This is a critical safety violation."
            )
        
        if not isinstance(df_after, pd.DataFrame):
            raise TypeError(
                f"❌ Variable '{self.df_name}' is no longer a DataFrame (type: {type(df_after).__name__}). "
                f"Transformation is BLOCKED."
            )
        
        final_shape = df_after.shape
        final_cols = set(df_after.columns)

        # ✅ Assert pour type checker
        assert self.initial_shape is not None, "initial_shape should be set in __enter__"
        assert self.initial_cols is not None, "initial_cols should be set in __enter__"

        # --- Vérification lignes ---
        if self.expect_rows == 'decrease' and final_shape[0] >= self.initial_shape[0]:
            raise AssertionError(
                f"❌ GUARD VIOLATION: Expected rows to decrease\n"
                f"   Before: {self.initial_shape[0]} rows\n"
                f"   After:  {final_shape[0]} rows"
            )
        elif self.expect_rows == 'same' and final_shape[0] != self.initial_shape[0]:
            raise AssertionError(
                f"❌ GUARD VIOLATION: Expected same number of rows\n"
                f"   Before: {self.initial_shape[0]} rows\n"
                f"   After:  {final_shape[0]} rows"
            )
        elif self.expect_rows == 'increase' and final_shape[0] <= self.initial_shape[0]:
            raise AssertionError(
                f"❌ GUARD VIOLATION: Expected rows to increase\n"
                f"   Before: {self.initial_shape[0]} rows\n"
                f"   After:  {final_shape[0]} rows"
            )

        # --- Vérification colonnes ---
        missing = self.initial_cols - final_cols
        added = final_cols - self.initial_cols
        
        missing_str = ', '.join(missing) if missing else 'None'
        added_str = ', '.join(added) if added else 'None'


        if self.expect_cols == 'same' and final_cols != self.initial_cols:
            raise AssertionError(f"❌ Columns changed! Missing: {missing or 'None'}, Added: {added or 'None'}")
        elif self.expect_cols == 'decrease' and len(final_cols) >= len(self.initial_cols):
            raise AssertionError(f"❌ Expected fewer columns: {len(self.initial_cols)} → {len(final_cols)}")
        elif self.expect_cols == 'increase' and len(final_cols) <= len(self.initial_cols):
            raise AssertionError(f"❌ Expected more columns: {len(self.initial_cols)} → {len(final_cols)}")

        # --- Verbose feedback ---
        if self.verbose:
            if final_shape == self.initial_shape and not (added or missing):
                msg = f"✅ {self.df_name}: Shape unchanged."
            else:
                msg = f"✅ {self.df_name}: {self.initial_shape} → {final_shape}"
        
            # Colonnes
            msg += f"\n   ➕ Added columns: {added_str}\n   ➖ Removed columns: {missing_str}"

            # Lignes
            row_diff = final_shape[0] - self.initial_shape[0]
            if row_diff > 0:
                msg += f"\n   ↗️  Rows increased by {row_diff} ({self.initial_shape[0]} → {final_shape[0]})"
            elif row_diff < 0:
                msg += f"\n   ↘️  Rows decreased by {-row_diff} ({self.initial_shape[0]} → {final_shape[0]})"
            else:
                msg += "\n   ↔️  Rows unchanged"

            print(msg)
        return False


# ✨ Raccourcis pratiques
VERBOSE_DEFAULT = True
def rows_decrease(df_name, verbose=VERBOSE_DEFAULT): return DataFrameGuard(df_name, expect_rows='decrease', verbose=verbose)
def rows_same(df_name, verbose=VERBOSE_DEFAULT): return DataFrameGuard(df_name, expect_rows='same', verbose=verbose)
def rows_increase(df_name, verbose=VERBOSE_DEFAULT): return DataFrameGuard(df_name, expect_rows='increase', verbose=verbose)

def cols_same(df_name, verbose=VERBOSE_DEFAULT): return DataFrameGuard(df_name, expect_cols='same', verbose=verbose)
def cols_decrease(df_name, verbose=VERBOSE_DEFAULT): return DataFrameGuard(df_name, expect_cols='decrease', verbose=verbose)
def cols_increase(df_name, verbose=VERBOSE_DEFAULT): return DataFrameGuard(df_name, expect_cols='increase', verbose=verbose)

def shape_same(df_name, verbose=VERBOSE_DEFAULT): 
    return DataFrameGuard(df_name, expect_rows='same', expect_cols='same', verbose=verbose)

