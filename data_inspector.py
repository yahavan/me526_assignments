"""
data_inspector.py
=================

A modular data sanitization and exploration engine for Google Colab.

Exposes two classes:

* ``DataInspector``  - end-to-end ingestion, cleaning, normalization,
  visualization and association analysis.
* ``PlottingMethods`` - granular, reusable chart builders (Bar, Pie,
  Histogram) that return HTML-wrapped Plotly figures for flexible
  embedding (dashboards, reports, emails, etc.).

Designed to be dropped into a Colab session and imported:

    >>> from data_inspector import DataInspector, PlottingMethods
    >>> insp = DataInspector()
    >>> insp.upload_data()          # Colab file picker
    >>> insp.data_summary()

Author: built for the "Modular Data Sanitization & Exploration Engine"
assignment.
"""

from __future__ import annotations

import io
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.stats as ss

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
)


# Strings that should be treated as missing values regardless of column.
DEFAULT_GARBAGE_TOKENS = [
    "?", "n/a", "na", "n.a.", "null", "none", "nan", "-", "--",
    "", " ", "missing", "unknown", "<na>",
]


class DataInspector:
    """Inspect, clean, normalize and visualize a tabular dataset.

    The active dataset is held in ``self.df``. Every mutating method
    updates ``self.df`` in place and returns ``self`` so calls can be
    chained, while every read-only method returns a fresh object and
    leaves ``self.df`` untouched.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        A dataframe to start from. If omitted, call :meth:`upload_data`
        or :meth:`load_csv` to populate the inspector.
    garbage_tokens : list of str, optional
        Case-insensitive string tokens to convert to ``NaN`` during
        sanitization. Defaults to a sensible built-in list.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        garbage_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        self.df: Optional[pd.DataFrame] = df.copy() if df is not None else None
        self.garbage_tokens = list(garbage_tokens) if garbage_tokens else list(
            DEFAULT_GARBAGE_TOKENS
        )

    # ------------------------------------------------------------------ #
    # Internal guards / helpers
    # ------------------------------------------------------------------ #
    def _has_data(self) -> bool:
        """Return ``True`` only if a non-empty dataframe is loaded."""
        if self.df is None:
            print("[DataInspector] No data loaded. Use upload_data() or load_csv() first.")
            return False
        if self.df.empty:
            print("[DataInspector] The dataset is empty - nothing to do.")
            return False
        return True

    @property
    def numeric_columns(self) -> list[str]:
        """List of numeric column names (empty if no data)."""
        if self.df is None:
            return []
        return self.df.select_dtypes(include=np.number).columns.tolist()

    @property
    def categorical_columns(self) -> list[str]:
        """List of non-numeric (object/category/bool) column names."""
        if self.df is None:
            return []
        return self.df.select_dtypes(exclude=np.number).columns.tolist()

    @staticmethod
    def _parse_csv_list(value: Union[str, Iterable, None]) -> list:
        """Parse comma-separated user input into a clean list.

        Accepts a raw string like ``"age, fare ,sex"`` or an iterable
        and returns ``['age', 'fare', 'sex']``. ``None`` -> ``[]``.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip() != ""]
        return [item for item in value]

    # ------------------------------------------------------------------ #
    # 1. Ingestion & sanitization
    # ------------------------------------------------------------------ #
    def upload_data(
        self,
        filepath: Optional[str] = None,
        sanitize: bool = True,
        auto_numeric: bool = True,
        **read_csv_kwargs,
    ) -> "DataInspector":
        """Ingest a CSV, from a Colab upload widget or a local path.

        In Google Colab, calling this with no arguments opens the file
        picker (``google.colab.files.upload``). Outside Colab, pass
        ``filepath`` to read from disk. After loading, the data is
        sanitized (garbage strings -> ``NaN``) and numeric columns are
        auto-detected unless disabled.

        Parameters
        ----------
        filepath : str, optional
            Path to a CSV file. If ``None`` the Colab uploader is used.
        sanitize : bool, default True
            Convert garbage tokens to ``NaN`` after loading.
        auto_numeric : bool, default True
            Force-convert object columns to numeric where appropriate.
        **read_csv_kwargs
            Extra keyword args forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        DataInspector
            ``self`` (for chaining).
        """
        if filepath is not None:
            self.df = pd.read_csv(filepath, **read_csv_kwargs)
        else:
            try:
                from google.colab import files  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "Colab upload is only available inside Google Colab. "
                    "Pass filepath='your.csv' when running locally."
                )
            uploaded = files.upload()
            if not uploaded:
                print("[DataInspector] No file was uploaded.")
                return self
            # Read the first uploaded file.
            name = next(iter(uploaded))
            self.df = pd.read_csv(io.BytesIO(uploaded[name]), **read_csv_kwargs)
            print(f"[DataInspector] Loaded '{name}'.")

        if sanitize:
            self.sanitize()
        if auto_numeric:
            self.auto_correct_types()
        return self

    def load_csv(self, filepath: str, **kwargs) -> "DataInspector":
        """Alias of :meth:`upload_data` for local files (no Colab needed)."""
        return self.upload_data(filepath=filepath, **kwargs)

    def sanitize(self) -> "DataInspector":
        """Replace garbage string tokens with real ``NaN`` values.

        Matching is case-insensitive and whitespace-trimmed, so ``' ? '``,
        ``'N/A'`` and ``'null'`` are all caught.
        """
        if not self._has_data():
            return self

        token_set = {t.strip().lower() for t in self.garbage_tokens}

        def _clean(value):
            if isinstance(value, str):
                if value.strip().lower() in token_set:
                    return np.nan
            return value

        obj_cols = self.df.select_dtypes(include="object").columns
        for col in obj_cols:
            self.df[col] = self.df[col].map(_clean)
        return self

    def auto_correct_types(self, numeric_threshold: float = 0.8) -> "DataInspector":
        """Force object columns to numeric where it makes sense.

        For each object column, a numeric coercion is attempted. The
        conversion is kept only if it does **not** produce an entirely
        null column *and* at least ``numeric_threshold`` of the originally
        non-null values survive the conversion. The threshold protects
        genuinely categorical columns (e.g. names, tickets) from being
        wiped out by coercion.

        Parameters
        ----------
        numeric_threshold : float, default 0.8
            Minimum fraction of originally non-null values that must
            convert successfully for the column to be treated as numeric.
        """
        if not self._has_data():
            return self

        for col in self.df.select_dtypes(include="object").columns:
            original = self.df[col]
            converted = pd.to_numeric(original, errors="coerce")

            non_null_before = original.notna().sum()
            non_null_after = converted.notna().sum()

            if non_null_after == 0:
                continue  # entirely null -> truly categorical, leave alone
            if non_null_before == 0:
                continue
            survival = non_null_after / non_null_before
            if survival >= numeric_threshold:
                self.df[col] = converted
        return self

    # ------------------------------------------------------------------ #
    # 2. Structural analysis & cleaning
    # ------------------------------------------------------------------ #
    def data_summary(self, preview_rows: int = 20) -> Optional[pd.DataFrame]:
        """Print a structural summary and return the head preview.

        Shows row/column counts, the numeric vs categorical split, a
        per-column missing-value breakdown and the first ``preview_rows``
        rows.

        Returns
        -------
        pandas.DataFrame or None
            The preview head (also displayed), or ``None`` if no data.
        """
        if not self._has_data():
            return None

        n_rows, n_cols = self.df.shape
        num_cols = self.numeric_columns
        cat_cols = self.categorical_columns

        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Rows:    {n_rows:,}")
        print(f"Columns: {n_cols:,}")
        print(f"Numeric ({len(num_cols)}):     {num_cols}")
        print(f"Categorical ({len(cat_cols)}): {cat_cols}")

        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        print("-" * 60)
        if missing.empty:
            print("Missing values: none")
        else:
            print("Missing values per column:")
            for col, count in missing.items():
                pct = 100 * count / n_rows
                print(f"  {col:<20} {count:>6}  ({pct:4.1f}%)")
        print("-" * 60)
        print(f"Duplicate rows: {self.df.duplicated().sum()}")
        print("=" * 60)

        preview = self.df.head(preview_rows)
        try:
            from IPython.display import display  # type: ignore
            display(preview)
        except Exception:
            print(preview)
        return preview

    def handle_missing_values(
        self,
        strategy: str = "mean",
        columns: Union[str, Sequence[str], None] = None,
        constant_value=None,
    ) -> "DataInspector":
        """Impute missing values using a chosen strategy.

        Parameters
        ----------
        strategy : {'mean', 'median', 'mode', 'constant'}, default 'mean'
            * ``mean`` / ``median`` apply to numeric columns only.
            * ``mode`` applies to any column (most frequent value).
            * ``constant`` fills with ``constant_value`` for any column.
        columns : str or list of str, optional
            Columns to operate on (comma-separated string accepted).
            Defaults to all eligible columns.
        constant_value : any, optional
            Required when ``strategy='constant'``.

        Returns
        -------
        DataInspector
            ``self`` (for chaining).
        """
        if not self._has_data():
            return self

        strategy = strategy.lower()
        valid = {"mean", "median", "mode", "constant"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{strategy}'.")

        cols = self._parse_csv_list(columns) or self.df.columns.tolist()
        cols = [c for c in cols if c in self.df.columns]

        for col in cols:
            if self.df[col].isna().sum() == 0:
                continue
            if strategy in ("mean", "median"):
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    continue  # skip non-numeric for mean/median
                fill = (
                    self.df[col].mean()
                    if strategy == "mean"
                    else self.df[col].median()
                )
            elif strategy == "mode":
                modes = self.df[col].mode(dropna=True)
                if modes.empty:
                    continue
                fill = modes.iloc[0]
            else:  # constant
                if constant_value is None:
                    raise ValueError("constant_value is required for strategy='constant'.")
                fill = constant_value
            self.df[col] = self.df[col].fillna(fill)
        return self

    def remove_duplicates(self, subset: Optional[Sequence[str]] = None) -> "DataInspector":
        """Drop exact duplicate rows (optionally limited to ``subset``)."""
        if not self._has_data():
            return self
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset).reset_index(drop=True)
        print(f"[DataInspector] Removed {before - len(self.df)} duplicate row(s).")
        return self

    def handle_outliers(
        self,
        columns: Union[str, Sequence[str], None] = None,
        action: str = "flag",
        multiplier: float = 1.5,
    ) -> "DataInspector":
        """Detect IQR-based outliers and either flag or delete them.

        An outlier is any value outside
        ``[Q1 - k*IQR, Q3 + k*IQR]`` where ``k`` is ``multiplier``.

        Parameters
        ----------
        columns : str or list of str, optional
            Numeric columns to check (comma-separated string accepted).
            Defaults to all numeric columns.
        action : {'flag', 'delete'}, default 'flag'
            * ``flag``   - add a boolean ``is_outlier`` column marking rows
              that are outliers in *any* checked column.
            * ``delete`` - drop those rows entirely.
        multiplier : float, default 1.5
            IQR multiplier (1.5 = standard Tukey fences).

        Returns
        -------
        DataInspector
            ``self`` (for chaining).
        """
        if not self._has_data():
            return self
        action = action.lower()
        if action not in {"flag", "delete"}:
            raise ValueError("action must be 'flag' or 'delete'.")

        cols = self._parse_csv_list(columns) or self.numeric_columns
        cols = [c for c in cols if c in self.numeric_columns]
        if not cols:
            print("[DataInspector] No numeric columns to check for outliers.")
            return self

        outlier_mask = pd.Series(False, index=self.df.index)
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            col_mask = (self.df[col] < lower) | (self.df[col] > upper)
            outlier_mask |= col_mask.fillna(False)

        n_out = int(outlier_mask.sum())
        if action == "flag":
            self.df["is_outlier"] = outlier_mask
            print(f"[DataInspector] Flagged {n_out} outlier row(s) in column 'is_outlier'.")
        else:
            self.df = self.df.loc[~outlier_mask].reset_index(drop=True)
            print(f"[DataInspector] Deleted {n_out} outlier row(s).")
        return self

    def delete_rows(self, indices: Union[str, Sequence[int]]) -> "DataInspector":
        """Delete rows by index from comma-separated input or a list.

        Example: ``delete_rows("0, 3, 17")`` drops those three rows.
        """
        if not self._has_data():
            return self
        raw = self._parse_csv_list(indices) if isinstance(indices, str) else list(indices)
        wanted = []
        for item in raw:
            try:
                wanted.append(int(item))
            except (ValueError, TypeError):
                print(f"[DataInspector] Skipping invalid row index: {item!r}")
        valid = [i for i in wanted if i in self.df.index]
        missing = set(wanted) - set(valid)
        if missing:
            print(f"[DataInspector] These indices were not found: {sorted(missing)}")
        self.df = self.df.drop(index=valid).reset_index(drop=True)
        print(f"[DataInspector] Deleted {len(valid)} row(s).")
        return self

    def delete_columns(self, columns: Union[str, Sequence[str]]) -> "DataInspector":
        """Delete columns by name from comma-separated input or a list.

        Example: ``delete_columns("ticket, cabin")``.
        """
        if not self._has_data():
            return self
        wanted = self._parse_csv_list(columns)
        valid = [c for c in wanted if c in self.df.columns]
        missing = [c for c in wanted if c not in self.df.columns]
        if missing:
            print(f"[DataInspector] These columns were not found: {missing}")
        self.df = self.df.drop(columns=valid)
        print(f"[DataInspector] Deleted {len(valid)} column(s): {valid}")
        return self

    # ------------------------------------------------------------------ #
    # 3. Feature engineering preparation (normalization)
    # ------------------------------------------------------------------ #
    def extract_normalized_numeric_data(
        self,
        method: str = "minmax",
        columns: Union[str, Sequence[str], None] = None,
    ) -> pd.DataFrame:
        """Return a scaled copy of the numeric columns.

        Parameters
        ----------
        method : {'minmax', 'standard', 'robust'}, default 'minmax'
            * ``minmax``   - scale each feature to [0, 1].
            * ``standard`` - Z-score (zero mean, unit variance).
            * ``robust``   - centre on the median, scale by the IQR.
        columns : str or list of str, optional
            Numeric columns to scale (comma-separated string accepted).

        Returns
        -------
        pandas.DataFrame
            Scaled numeric data (empty frame if no data/columns).
        """
        if not self._has_data():
            return pd.DataFrame()

        cols = self._parse_csv_list(columns) or self.numeric_columns
        cols = [c for c in cols if c in self.numeric_columns]
        if not cols:
            print("[DataInspector] No numeric columns available to normalize.")
            return pd.DataFrame()

        scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }
        if method not in scalers:
            raise ValueError(f"method must be one of {set(scalers)}, got '{method}'.")

        subset = self.df[cols]
        # Scalers do not accept NaN; impute with the column median first.
        filled = subset.fillna(subset.median(numeric_only=True))
        scaled = scalers[method].fit_transform(filled)
        return pd.DataFrame(scaled, columns=cols, index=self.df.index)

    def extract_normalized_categorical_data(
        self,
        method: str = "onehot",
        columns: Union[str, Sequence[str], None] = None,
    ) -> pd.DataFrame:
        """Return an encoded copy of the categorical columns.

        Parameters
        ----------
        method : {'onehot', 'ordinal', 'uniform'}, default 'onehot'
            * ``onehot``  - one binary column per category.
            * ``ordinal`` - integer code per category.
            * ``uniform`` - ordinal codes rescaled to [0, 1].
        columns : str or list of str, optional
            Categorical columns to encode (comma-separated string accepted).

        Returns
        -------
        pandas.DataFrame
            Encoded categorical data (empty frame if no data/columns).
        """
        if not self._has_data():
            return pd.DataFrame()

        cols = self._parse_csv_list(columns) or self.categorical_columns
        cols = [c for c in cols if c in self.categorical_columns]
        if not cols:
            print("[DataInspector] No categorical columns available to encode.")
            return pd.DataFrame()

        if method not in {"onehot", "ordinal", "uniform"}:
            raise ValueError("method must be 'onehot', 'ordinal' or 'uniform'.")

        # Fill missing categories with an explicit token so encoders are stable.
        subset = self.df[cols].astype("object").fillna("__missing__")

        if method == "onehot":
            return pd.get_dummies(subset, columns=cols, dtype=int)

        encoder = OrdinalEncoder()
        encoded = encoder.fit_transform(subset)
        encoded_df = pd.DataFrame(encoded, columns=cols, index=self.df.index)

        if method == "uniform":
            for col in cols:
                n_categories = len(encoder.categories_[cols.index(col)])
                denom = max(n_categories - 1, 1)
                encoded_df[col] = encoded_df[col] / denom
        return encoded_df

    def merge_processed_data(
        self,
        numeric_method: str = "standard",
        categorical_method: str = "onehot",
    ) -> pd.DataFrame:
        """Build a single model-ready frame: scaled numerics + encoded cats.

        Parameters
        ----------
        numeric_method : {'minmax', 'standard', 'robust'}, default 'standard'
        categorical_method : {'onehot', 'ordinal', 'uniform'}, default 'onehot'

        Returns
        -------
        pandas.DataFrame
            Concatenated, fully numeric feature matrix.
        """
        if not self._has_data():
            return pd.DataFrame()

        numeric = self.extract_normalized_numeric_data(method=numeric_method)
        categorical = self.extract_normalized_categorical_data(method=categorical_method)
        merged = pd.concat([numeric, categorical], axis=1)
        return merged

    # ------------------------------------------------------------------ #
    # Association statistics (helpers)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Bias-corrected Cramer's V for two categorical series."""
        confusion = pd.crosstab(x, y)
        if confusion.shape[0] < 2 or confusion.shape[1] < 2:
            return np.nan
        chi2 = ss.chi2_contingency(confusion, correction=False)[0]
        n = confusion.to_numpy().sum()
        if n == 0:
            return np.nan
        phi2 = chi2 / n
        r, k = confusion.shape
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return np.nan
        return float(np.sqrt(phi2corr / denom))

    @staticmethod
    def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
        """Eta (correlation ratio) between a categorical and a numeric series.

        Generalizes point-biserial correlation to any number of groups
        and is computed from between-group vs total variance (one-way
        ANOVA). Ranges from 0 (no association) to 1.
        """
        frame = pd.DataFrame({"cat": categories, "val": values}).dropna()
        if frame.empty or frame["cat"].nunique() < 2:
            return np.nan
        grouped = frame.groupby("cat")["val"]
        counts = grouped.count()
        means = grouped.mean()
        grand_mean = frame["val"].mean()
        ss_between = float((counts * (means - grand_mean) ** 2).sum())
        ss_total = float(((frame["val"] - grand_mean) ** 2).sum())
        if ss_total == 0:
            return 0.0
        return float(np.sqrt(ss_between / ss_total))

    # ------------------------------------------------------------------ #
    # 4. Interactive visualization (Plotly)
    # ------------------------------------------------------------------ #
    def plot_univariate(self, column: str):
        """Three-panel univariate view of a numeric column.

        Panels (left to right): horizontal box/violin, scatter of index
        vs value, and histogram.

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        if not self._has_data():
            return None
        if column not in self.numeric_columns:
            print(f"[DataInspector] '{column}' is not a numeric column.")
            return None

        series = self.df[column].dropna()
        if series.empty:
            print(f"[DataInspector] '{column}' has no non-null values to plot.")
            return None

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Distribution (Box/Violin)", "Index vs Value", "Histogram"),
            horizontal_spacing=0.08,
        )
        fig.add_trace(
            go.Violin(x=series, name=column, box_visible=True,
                      meanline_visible=True, orientation="h", points="outliers"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, mode="markers",
                       name="value", marker=dict(size=5, opacity=0.6)),
            row=1, col=2,
        )
        fig.add_trace(
            go.Histogram(x=series, name="count", nbinsx=30),
            row=1, col=3,
        )
        fig.update_layout(
            title_text=f"Univariate analysis: '{column}'",
            showlegend=False,
            height=420,
            template="plotly_white",
        )
        return fig

    def plot_relationship(self, x: str, y: str):
        """Auto-pick the right chart for the relationship between two columns.

        * numeric vs numeric -> scatter with OLS trendline.
        * categorical vs numeric -> box plot showing all points.
        * categorical vs categorical -> grouped bar chart.

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        if not self._has_data():
            return None
        for col in (x, y):
            if col not in self.df.columns:
                print(f"[DataInspector] Column '{col}' not found.")
                return None

        x_is_num = x in self.numeric_columns
        y_is_num = y in self.numeric_columns
        data = self.df[[x, y]].dropna()
        if data.empty:
            print("[DataInspector] No overlapping non-null rows to plot.")
            return None

        if x_is_num and y_is_num:
            # statsmodels is required for the OLS trendline; degrade gracefully.
            trendline = None
            try:
                import statsmodels.api  # noqa: F401
                trendline = "ols"
            except ImportError:
                print("[DataInspector] statsmodels not found - drawing scatter without trendline.")
            fig = px.scatter(data, x=x, y=y, trendline=trendline,
                             opacity=0.65, template="plotly_white",
                             title=f"{y} vs {x} (numeric - numeric)")
        elif x_is_num != y_is_num:
            cat_col, num_col = (x, y) if not x_is_num else (y, x)
            fig = px.box(data, x=cat_col, y=num_col, points="all",
                         template="plotly_white",
                         title=f"{num_col} by {cat_col} (categorical - numeric)")
        else:
            grouped = data.groupby([x, y]).size().reset_index(name="count")
            fig = px.bar(grouped, x=x, y="count", color=y, barmode="group",
                         template="plotly_white",
                         title=f"{x} vs {y} (categorical - categorical)")
        fig.update_layout(height=480)
        return fig

    def plot_categorical_frequency(self, column: str, top_n: int = 20):
        """Bar chart of category counts annotated with percentage labels.

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        if not self._has_data():
            return None
        if column not in self.df.columns:
            print(f"[DataInspector] Column '{column}' not found.")
            return None

        counts = self.df[column].value_counts(dropna=False).head(top_n)
        if counts.empty:
            print(f"[DataInspector] '{column}' has no values to plot.")
            return None

        total = counts.sum()
        pct = (counts / total * 100).round(1)
        labels = [f"{c} ({p}%)" for c, p in zip(counts.values, pct.values)]

        fig = go.Figure(
            go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                text=labels,
                textposition="outside",
                marker_color="#4C78A8",
            )
        )
        fig.update_layout(
            title=f"Frequency of '{column}'",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            height=440,
        )
        return fig

    # ------------------------------------------------------------------ #
    # 5. Deep statistical insights
    # ------------------------------------------------------------------ #
    def compute_association_matrix(self) -> pd.DataFrame:
        """Return a unified association matrix across all column types.

        Cell values use the metric appropriate to the pair of columns:
        Pearson |r| for numeric-numeric, Cramer's V for
        categorical-categorical, and the correlation ratio (eta) for
        mixed pairs. All metrics share a 0-1 scale where higher means a
        stronger association.
        """
        if not self._has_data():
            return pd.DataFrame()

        cols = self.df.columns.tolist()
        num_set = set(self.numeric_columns)
        matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                a_num, b_num = a in num_set, b in num_set
                if a_num and b_num:
                    val = self.df[[a, b]].corr().iloc[0, 1]
                    val = abs(val) if pd.notna(val) else np.nan
                elif (not a_num) and (not b_num):
                    val = self._cramers_v(self.df[a], self.df[b])
                else:
                    cat, num = (a, b) if not a_num else (b, a)
                    val = self._correlation_ratio(self.df[cat], self.df[num])
                matrix.loc[a, b] = val
                matrix.loc[b, a] = val
        return matrix

    def plot_all_associations_heatmap(self):
        """Heatmap of the unified association matrix (see
        :meth:`compute_association_matrix`).

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        if not self._has_data():
            return None
        matrix = self.compute_association_matrix()
        if matrix.empty:
            return None

        fig = px.imshow(
            matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=0,
            zmax=1,
            aspect="auto",
            title="Unified associations  (|Pearson| / Cramer's V / Eta)",
        )
        fig.update_layout(height=620, template="plotly_white")
        return fig


class PlottingMethods:
    """Granular, reusable chart builders returning HTML-wrapped figures.

    Each method returns a self-contained HTML ``str`` (via
    ``Figure.to_html``) so charts can be embedded into dashboards,
    notebooks, reports or emails independently of the inspector.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        A default dataframe so column names can be passed directly to the
        chart methods. Series/arrays may also be passed without a df.
    include_plotlyjs : str, default 'cdn'
        Forwarded to ``Figure.to_html``. ``'cdn'`` keeps the HTML small;
        use ``True`` to inline the full Plotly library for offline use.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, include_plotlyjs: str = "cdn") -> None:
        self.df = df.copy() if df is not None else None
        self.include_plotlyjs = include_plotlyjs

    def _resolve(self, data, column: Optional[str]) -> pd.Series:
        """Resolve a Series from an explicit ``data`` arg or a df column."""
        if data is not None:
            return pd.Series(data).dropna()
        if column is not None and self.df is not None and column in self.df.columns:
            return self.df[column].dropna()
        raise ValueError("Provide either `data` or a valid `column` (with a df set).")

    def _to_html(self, fig) -> str:
        """Wrap a Plotly figure as an embeddable HTML fragment."""
        return fig.to_html(full_html=False, include_plotlyjs=self.include_plotlyjs)

    def bar_chart(self, data=None, column: Optional[str] = None,
                  title: str = "Bar Chart", as_html: bool = True):
        """Bar chart of value counts. Returns HTML (default) or a Figure."""
        series = self._resolve(data, column)
        if series.empty:
            return "" if as_html else None
        counts = series.value_counts()
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values,
                               marker_color="#4C78A8"))
        fig.update_layout(title=title, template="plotly_white",
                          xaxis_title=column or "category", yaxis_title="count")
        return self._to_html(fig) if as_html else fig

    def pie_chart(self, data=None, column: Optional[str] = None,
                  title: str = "Pie Chart", as_html: bool = True):
        """Pie chart of category proportions. Returns HTML or a Figure."""
        series = self._resolve(data, column)
        if series.empty:
            return "" if as_html else None
        counts = series.value_counts()
        fig = go.Figure(go.Pie(labels=counts.index.astype(str), values=counts.values,
                               hole=0.3))
        fig.update_layout(title=title, template="plotly_white")
        return self._to_html(fig) if as_html else fig

    def histogram(self, data=None, column: Optional[str] = None,
                  bins: int = 30, title: str = "Histogram", as_html: bool = True):
        """Histogram of a numeric series. Returns HTML or a Figure."""
        series = pd.to_numeric(self._resolve(data, column), errors="coerce").dropna()
        if series.empty:
            return "" if as_html else None
        fig = go.Figure(go.Histogram(x=series, nbinsx=bins, marker_color="#54A24B"))
        fig.update_layout(title=title, template="plotly_white",
                          xaxis_title=column or "value", yaxis_title="count")
        return self._to_html(fig) if as_html else fig
