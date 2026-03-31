import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Literal, overload
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
    max_error,
)


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


def export_use_type_summary(df, columns, output_path, overwrite=False):
    """
    Create a CSV summary of unique building use types by column.
    For comma-separated columns, only the first value before the comma is counted.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing building use columns.
    columns : list[str]
        Column names to analyze.
    output_path : str or Path
        CSV output path.
    """

    if not overwrite and os.path.exists(output_path):
        print(f"✅ Fichier déjà existant : {output_path}")
        return

    results = []

    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found, skipping.")
            continue

        # Récupération de la colonne, suppression des NaN
        series = df[col].dropna().astype(str)

        # Si la colonne contient des listes séparées par virgule, ne garder que la première
        if series.str.contains(',').any():
            first_values = series.str.split(',', n=1).str[0].str.strip()
        else:
            first_values = series.str.strip()

        # Comptage des occurrences
        counts = first_values.value_counts().sort_index()

        # Construction du DataFrame temporaire
        temp_df = pd.DataFrame({
            "df_column": col,
            "use_type": counts.index,
            "count": counts.values
        })

        results.append(temp_df)

    # Fusionner tous les résultats
    summary_df = pd.concat(results, ignore_index=True)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(f"✅ Exported {len(summary_df)} rows to {output_path}")
    return summary_df

def analyse_cardinality(df, simple_features, multi_features=None, separator=',', plot=False, figsize=(8,4), color='skyblue'):
    """
    Calcule la cardinalité pour des colonnes single-label et multi-label,
    avec option d'afficher un barplot annoté.

    Parameters:
        df : pd.DataFrame
        simple_features : list -> colonnes à traiter comme single-label
        multi_features : list -> colonnes à traiter comme multi-label (split + explode)
        separator : str -> séparateur pour les colonnes multi-label
        plot : bool -> si True, affiche un barplot annoté
        figsize : tuple -> taille du plot
        color : str -> couleur des barres du plot

    Returns:
        pd.DataFrame avec les colonnes 'column' et 'cardinality'
    """
    multi_features = multi_features or []
    all_cols = simple_features + multi_features

    def cardinality(col_series, multi=False):
        s = col_series.dropna()
        if multi:
            return s.str.split(separator).explode().str.strip().nunique()
        else:
            return s.nunique()

    df_analyse = pd.DataFrame({
        "column": all_cols,
        "cardinality": [cardinality(df[col], multi=(col in multi_features)) for col in all_cols]
    })

    if plot:
        plt.figure(figsize=figsize)
        bars = plt.barh(df_analyse['column'], df_analyse['cardinality'], color=color)
        plt.xlabel("Cardinalité")
        plt.ylabel("Colonne")
        plt.title("Cardinalité des colonnes")
        plt.gca().invert_yaxis()

        # ➕ Ajout des annotations sur chaque barre
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + max(df_analyse['cardinality']) * 0.01,  # petit décalage à droite
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va='center',
                fontsize=9,
                color='black'
            )

        plt.tight_layout()
        plt.show()

    return df_analyse



@overload
def impute_other(df: pd.DataFrame, verbose: bool = False, return_changes: Literal[False] = False) -> pd.DataFrame: ...

@overload
def impute_other(df: pd.DataFrame, verbose: bool = False, return_changes: Literal[True] = True) -> Tuple[pd.DataFrame, pd.DataFrame]: ...

def impute_other(df, verbose=False, return_changes=False): 

    df = df.copy()

    mask_other = df["LargestPropertyUseType"].eq("Other")

    # Préparation des sources potentielles
    primary_ok = (
        df["PrimaryPropertyType"].notna()
        & ~df["PrimaryPropertyType"].eq("Mixed Use Property")
    )
    first_use = df["ListOfAllPropertyUseTypes"].astype(str).str.split(",").str[0].str.strip()
    list_ok = ~first_use.eq("Other") & first_use.ne("nan")  # évite les 'nan' string

    # Création d’une colonne d’imputation prioritaire
    new_values = df["LargestPropertyUseType"].copy()
    source_col = pd.Series([None] * len(df), index=df.index)

    # Imputation prioritaire : PrimaryPropertyType, sinon ListOfAllPropertyUseTypes
    new_values.loc[mask_other & primary_ok] = df.loc[mask_other & primary_ok, "PrimaryPropertyType"]
    source_col.loc[mask_other & primary_ok] = "PrimaryPropertyType"

    remaining_mask = mask_other & ~primary_ok & list_ok
    new_values.loc[remaining_mask] = first_use.loc[remaining_mask]
    source_col.loc[remaining_mask] = "ListOfAllPropertyUseTypes"

    # Détection des modifications
    changed = new_values != df["LargestPropertyUseType"]
    df["LargestPropertyUseType"] = new_values

    modified_rows = df.loc[changed, ["PropertyName", "LargestPropertyUseType"]].assign(
        OldValue="Other",
        NewValue=df.loc[changed, "LargestPropertyUseType"],
        Source=source_col.loc[changed].values
    ).reset_index(names="index")

    if verbose:
        print(f"\n🔍 {len(modified_rows)} lignes imputées sur {mask_other.sum()} 'Other'")
        if not modified_rows.empty:
            print(modified_rows.to_string(index=False))
        else:
            print("Aucune modification effectuée.")

    if return_changes:
        return df, modified_rows

    return df

def get_SiteEnergyUse(df):
    return (df["Electricity(kWh)"] * 3.412) + (df["NaturalGas(therms)"] *100) + df["SteamUse(kBtu)"]


def get_SiteEUI(df):
    return ( df["SiteEnergyUse(kBtu)(kWh)"] / (df["PropertyGFABuilding"]) )

def get_sparsity_report(df, round_digits=2):
    """
    Retourne un DataFrame indiquant le pourcentage de zéros par feature.
    """
    return (
        (1 - df.mean())                     # Ratio de zéros
        .mul(100)                           # En pourcentage
        .round(round_digits)                # Arrondi
        .sort_values(ascending=False)       # Tri décroissant
        .reset_index(name="zero_rate")      # Transforme en DataFrame
        .rename(columns={"index": "features"})  # Renomme la colonne d'index
    )

def filter_by_malus_level(
    df: pd.DataFrame,
    malus_col: str = "malus_score",
    level: int = 0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Filtre le DataFrame selon un niveau de tolérance de malus.
    Plus le niveau augmente, plus le filtrage est souple.

    Parameters
    ----------
    df : pd.DataFrame
        Jeu de données à filtrer.
    malus_col : str, default='malus_score'
        Colonne contenant les scores de malus cumulés.
    level : int, default=0
        Niveau de tolérance (chaque incrément relâche le seuil).
    verbose : bool, default=True
        Si True, affiche le nombre de lignes conservées.

    Returns
    -------
    pd.DataFrame
        Sous-ensemble filtré du DataFrame original.
    """
    
    if malus_col not in df.columns:
        raise KeyError(f"La colonne '{malus_col}' est absente du DataFrame.")
    
    if not np.issubdtype(df[malus_col].dtype, np.number):
        raise TypeError(f"La colonne '{malus_col}' doit être numérique.")

    bonus_max = df[malus_col].max()
    threshold = bonus_max - level

    mask = df[malus_col] < threshold
    filtered = df.loc[mask, :]

    if verbose:
        retained = mask.sum()
        total = len(df)
        pct = retained / total * 100
        print(f"✅ Level {level} → seuil < {threshold} | {retained}/{total} lignes conservées ({pct:.1f}%) — {total - retained} supprimées")


    return filtered


def regression_metrics_frame(
    y_true_test,
    y_pred_test,
    y_true_train=None,
    y_pred_train=None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return regression metrics in a tidy DataFrame.

    Parameters
    ----------
    y_true_test, y_pred_test : array-like
        Ground truth and predictions for the test split.
    y_true_train, y_pred_train : array-like, optional
        Ground truth and predictions for the train split. When both are
        provided, train metrics are computed as well.
    verbose : bool, default=False
        If ``True``, pretty-prints the computed metrics and basic test error
        statistics.

    Returns
    -------
    pd.DataFrame
        A dataframe indexed by metric name with two columns: ``train`` and
        ``test``.

    Notes
    -----
    When ``verbose`` is ``True`` and train metrics are available, the helper
    emits an overfitting warning if the train R² exceeds the test R² by more
    than 0.15.

    Raises
    ------
    ValueError
        If only one of ``y_true_train`` or ``y_pred_train`` is provided.
    """

    overfit_threshold = 0.15

    metric_functions = {
        "R2": r2_score,
        "MAE": mean_absolute_error,
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE(%)": lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MedianAE": median_absolute_error,
        "ExplainedVar": explained_variance_score,
        "MaxError": max_error,
    }

    def _compute_metrics(y_true, y_pred):
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        return {
            name: func(y_true_arr, y_pred_arr)
            for name, func in metric_functions.items()
        }

    test_metrics = _compute_metrics(y_true_test, y_pred_test)

    has_train_targets = y_true_train is not None
    has_train_predictions = y_pred_train is not None

    if has_train_targets != has_train_predictions:
        raise ValueError(
            "Both 'y_true_train' and 'y_pred_train' are required to compute train metrics."
        )

    include_train = has_train_targets and has_train_predictions
    if include_train:
        train_metrics = _compute_metrics(y_true_train, y_pred_train)
    else:
        train_metrics = {name: np.nan for name in metric_functions}

    metrics_df = pd.DataFrame({
        "train": pd.Series(train_metrics),
        "test": pd.Series(test_metrics),
    }).loc[list(metric_functions.keys()), ["train", "test"]]

    if verbose:
        def _format_metrics(metrics):
            return "\n".join(
                f"{metric:12s}: {value:,.4f}" if isinstance(value, (int, float, np.floating)) else f"{metric:12s}: {value}"
                for metric, value in metrics.items()
            )

        _emit(verbose, "result", "=== METRICS TEST ===\n" + _format_metrics(test_metrics))

        if include_train:
            _emit(verbose, "result", "=== METRICS TRAIN ===\n" + _format_metrics(train_metrics))

            r2_gap = train_metrics["R2"] - test_metrics["R2"]
            if r2_gap > overfit_threshold:
                _emit(
                    verbose,
                    "warning",
                    (
                        "Possible overfitting detected: Train R2 "
                        f"({train_metrics['R2']:.4f}) exceeds Test R2 "
                        f"({test_metrics['R2']:.4f}) by {r2_gap:.4f} (threshold {overfit_threshold:.2f})."
                    ),
                )
        else:
            _emit(
                verbose,
                "info",
                "=== METRICS TRAIN ===\nTrain metrics not computed (provide both 'y_true_train' and 'y_pred_train').",
            )

        errors = np.asarray(y_true_test) - np.asarray(y_pred_test)
        _emit(
            verbose,
            "info",
            "Erreur (y_true - y_pred) — test :\n"
            + f"  mean: {errors.mean():,.4f} | std: {errors.std():,.4f} | median: {np.median(errors):,.4f}",
        )

    return metrics_df


def plot_target_distribution_comparison(
    df_before, df_after, cols,
    title_suffix="(avant/après nettoyage)",
    log_scale=False,
    show=True
):
    """
    Compare la distribution des colonnes avant et après nettoyage.

    Args:
        df_before (pd.DataFrame): Données avant nettoyage
        df_after (pd.DataFrame): Données après nettoyage
        cols (list[str]): Colonnes à comparer
        title_suffix (str): Texte ajouté au titre principal
        log_scale (bool): Si True, applique une échelle log
        show (bool): Si True, affiche la figure directement

    Returns:
        (fig, axes): objets Matplotlib pour personnalisation ultérieure
    """
    n_cols = len(cols)
    fig, axes = plt.subplots(n_cols, 2, figsize=(18, 6 * n_cols))
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        # --- Histogrammes avant / après ---
        sns.histplot(df_before[col], color="gray", kde=True, label="Avant", ax=axes[i][0], alpha=0.5)
        sns.histplot(df_after[col], color="steelblue", kde=True, label="Après", ax=axes[i][0], alpha=0.6)
        axes[i][0].set_title(f"Distribution de {col} {title_suffix}", fontsize=14)
        axes[i][0].set_xlabel(col)
        axes[i][0].set_ylabel("Nombre d'observations")
        axes[i][0].legend()

        if log_scale:
            axes[i][0].set_xscale("log")

        # --- Boxplots côte à côte (compatible Seaborn ≥ 0.14)
        df_box = pd.concat([
            pd.DataFrame({col: df_before[col], "dataset": "Avant"}),
            pd.DataFrame({col: df_after[col], "dataset": "Après"})
        ])
        sns.boxplot(
            data=df_box,
            x="dataset",
            y=col,
            hue="dataset",
            palette=["gray", "steelblue"],
            ax=axes[i][1],
            legend=False
        )
        axes[i][1].set_title(f"Boxplot de {col} {title_suffix}", fontsize=14)
        axes[i][1].set_xlabel("")
        axes[i][1].set_ylabel(col)

        if log_scale:
            axes[i][1].set_yscale("log")

        # --- Statistiques résumées (moyenne, médiane) ---
        mean_b, mean_a = df_before[col].mean(), df_after[col].mean()
        med_b, med_a = df_before[col].median(), df_after[col].median()
        axes[i][0].text(
            0.98, 0.95,
            f"μ avant: {mean_b:,.0f}\nμ après: {mean_a:,.0f}\n"
            f"med avant: {med_b:,.0f}\nmed après: {med_a:,.0f}",
            transform=axes[i][0].transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

    # --- Légende globale (une seule fois) ---
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    # --- Titre global ---
    rows_diff = len(df_before) - len(df_after)
    fig.suptitle(
        f"Comparaison des distributions {title_suffix} – {rows_diff} lignes supprimées",
        fontsize=18
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if show:
        plt.show()

    return fig, axes


def get_binary_features(df: pd.DataFrame, verbose: bool = False) -> list[str]:
    """Return numeric columns whose non-null values are constrained to ``{0, 1}``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing candidate columns.
    verbose : bool, default=True
        If ``True``, reports diagnostic information via ``_emit``.

    Returns
    -------
    list[str]
        Ordered column names whose observed values are strictly binary.
    """

    candidate_df = df.select_dtypes(include=[np.number, "boolean"])

    candidate_columns = list(candidate_df.columns)
    _emit(
        verbose,
        "info",
        f"Scanning {len(candidate_columns)} column(s) for strictly binary values.",
    )

    binary_columns: list[str] = []

    for column in candidate_columns:
        series = candidate_df[column].dropna()
        if series.empty:
            continue

        unique_values = set(pd.unique(series))
        normalized_values = {float(value) for value in unique_values}

        if normalized_values <= {0.0, 1.0}:
            binary_columns.append(column)

    if binary_columns:
        preview = ", ".join(binary_columns[:5])
        remainder = "" if len(binary_columns) <= 5 else f" (+{len(binary_columns) - 5} more)"
        _emit(
            verbose,
            "result",
            f"Identified {len(binary_columns)} strictly binary column(s): {preview}{remainder}",
        )
    else:
        _emit(verbose, "result", "No strictly binary columns detected.")

    return binary_columns


def get_features_to_scale(
    df: pd.DataFrame,
    exclude_binary: bool = True,
    verbose: bool = False,
) -> list[str]:
    """Return numeric feature names suitable for scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing candidate features.
    exclude_binary : bool, default=True
        When ``True``, drops columns whose non-null values are restricted to
        ``{0, 1}`` (after coercing boolean values to integers) from the result.

    verbose : bool, default=True
        If ``True``, reports diagnostic information via ``_emit``.

    Returns
    -------
    list[str]
        Ordered column names that should be passed to a scaler.
    """

    numeric_df = df.select_dtypes(include=[np.number])

    numeric_columns = list(numeric_df.columns)
    _emit(
        verbose,
        "info",
        f"Identified {len(numeric_columns)} numeric column(s) out of {df.shape[1]} total.",
    )

    if not numeric_columns:
        _emit(verbose, "result", "No numeric columns available for scaling.")
        return []

    if not exclude_binary:
        _emit(
            verbose,
            "result",
            f"Returning all {len(numeric_columns)} numeric column(s) for scaling.",
        )
        return numeric_columns

    binary_columns = get_binary_features(df, verbose=verbose)
    binary_set = {column for column in binary_columns if column in numeric_columns}
    binary_columns = sorted(binary_set, key=numeric_columns.index)
    scale_columns = [column for column in numeric_columns if column not in binary_set]

    if binary_columns:
        _emit(
            verbose,
            "update",
            f"Excluded {len(binary_columns)} strictly binary column(s) from scaling.",
        )

    _emit(
        verbose,
        "result",
        f"Returning {len(scale_columns)} column(s) ready for scaling.",
    )
    return scale_columns


class DataFrameInspector(BaseEstimator, TransformerMixin):
    def __init__(self, show_cols=True, show_shape=True, message=None):
        self.show_cols = show_cols
        self.show_shape = show_shape
        self.message = message

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.message:
            print(f"\n🔍 {self.message}")
        if self.show_shape:
            print(f"Shape: {X.shape}")
        if self.show_cols:
            print(f"Colonnes: {list(X.columns)}")
        return X  # Important : on renvoie X inchangé