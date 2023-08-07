from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib


def make_table(df, datasets, optimizers, schedules, metric="accuracy", mode="max"):
    table = df[df["dataset"].isin(datasets)]
    table = table[
        ["dataset", "optimizer", "schedule", f"{metric}_mean", f"{metric}_std"]
    ]
    metrics = [(dataset, metric) for dataset in datasets]

    table = col_to_header(table, new_header="dataset", index=["optimizer", "schedule"])
    new_index = []
    for optimizer in optimizers:
        for schedule in schedules:
            key = (optimizer, schedule)
            if key in table.index:
                new_index.append(key)
    table = table.reindex(index=new_index)
    mask_best = get_best_within_std(table, params_fixed=[], metrics=metrics, modes=mode)

    table = merge_mean_std(table, metrics, formats="{:.2f}")
    style = table.style
    style = apply_style_attribute(style, mask=mask_best)
    return style


def make_topk_tables(
    df, datasets, optimizers, schedules, ks, metric="accuracy", mode="max", suffix=""
):
    for k in ks:
        mean_topk = get_topk_results(
            df,
            other_variables=["dataset", "optimizer", "schedule"],
            top_variable="base_lr",
            k=k,
        )
        style = make_table(
            df=mean_topk,
            datasets=datasets,
            optimizers=optimizers,
            schedules=schedules,
            metric=metric,
            mode=mode,
        )
        style.to_latex(f"../pub/tables/top{k}_{suffix}.tex", hrules=True)


def filter_dataframe(dataframe, filters):
    """
    Filter a pandas DataFrame based on a list of dictionaries.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to be filtered.
        filters (list of dict): List of dictionaries representing the filters.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = dataframe.copy()

    filter_query = " or ".join(
        [
            "("
            + " and ".join(
                [f"{key} == {repr(value)}" for key, value in filter_dict.items()]
            )
            + ")"
            for filter_dict in filters
        ]
    )

    filtered_df = filtered_df.query(filter_query)
    return filtered_df


def agg_dataframe(data, cols_to_keep, agg_fns=["mean", "std"]):
    # Get names of numerical columns
    numerical_cols = data.select_dtypes("number").columns
    numerical_cols_excl = (
        cols_to_keep if "seed" in cols_to_keep else cols_to_keep + ["seed"]
    )
    numerical_cols = numerical_cols.drop(labels=numerical_cols_excl, errors="ignore")

    # Aggregate metrics over all variables except seed
    data_mean = data.groupby(numerical_cols_excl)[numerical_cols].mean().reset_index()

    # Aggregate metrics over random seed
    data_mean = data_mean.groupby(cols_to_keep).agg(func=agg_fns)
    data_mean.columns = data_mean.columns.to_flat_index()
    data_mean.columns = [f"{col[0]}_{col[1]}" for col in data_mean.columns]
    data_mean.reset_index(inplace=True)
    return data_mean


def get_best_params(data, params, fixed_params, metric, mode="max"):
    if isinstance(params, str):
        params = [params]

    # Get names of numerical columns
    numerical_cols = data.select_dtypes("number").columns
    numerical_cols = numerical_cols.drop(
        labels=fixed_params + params + ["seed"], errors="ignore"
    )

    data_mean = data.groupby(fixed_params + params)[numerical_cols].mean().reset_index()
    idcs_best = (
        data_mean.groupby(fixed_params)[metric].idxmax()
        if mode == "max"
        else data_mean.groupby(fixed_params)[metric].idxmin()
    )
    best_params = data_mean.loc[idcs_best, params]
    best_params.index = idcs_best.index
    result = best_params.reset_index().merge(data, how="inner")
    return result


class ValueColorMapper:
    def __init__(self, values, cmap="viridis") -> None:
        self.ncolors = len(values)

        cmap = matplotlib.colormaps[cmap]
        cnorm = matplotlib.colors.Normalize(vmin=0, vmax=self.ncolors - 1)
        self.mapper = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
        self.codec = dict(zip(values, np.arange(self.ncolors)))

    def __call__(self, value):
        return self.mapper.to_rgba(self.codec[value])


def make_parallel_subplots(
    data,
    var_cols,
    var_colors,
    var_x,
    vars_y,
    x_scale="linear",
    y_scale="linear",
    axsize=(6, 3),
    sharex="col",
    smoothing=None,
    cmap="viridis",
):
    param_configs = set()
    cmapper = ValueColorMapper(data[var_colors].unique(), cmap=cmap)
    n_rows = len(vars_y)
    n_cols = data[var_cols].nunique()
    if smoothing is None:
        smoothing = n_rows * [1]
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(axsize[0] * n_cols, axsize[1] * n_rows),
        sharex=sharex,
    )
    for idx_col, (name_col, data_col) in enumerate(data.groupby(var_cols)):
        for key_color, data_color in data_col.groupby(var_colors):
            for idx_row, var_y in enumerate(vars_y):
                data_color = data_color.sort_values(by=var_x)
                ax = axs[idx_row, idx_col]
                ax.set_xscale(x_scale)
                ax.set_yscale(y_scale)
                if idx_row == 0:
                    ax.set_title(name_col)
                elif idx_row == len(vars_y) - 1:
                    ax.set_xlabel(var_x)
                if idx_col == 0:
                    ax.set_ylabel(var_y)
                ax.plot(
                    data_color[var_x],
                    data_color[var_y].ewm(alpha=smoothing[idx_row]).mean(),
                    label=key_color,
                    c=cmapper(key_color),
                )
                params = (name_col, key_color, var_y)
                if params in param_configs:
                    print("double loop!")
                param_configs.add((name_col, key_color, var_y))
    axs[-1, -1].legend(loc=(1.04, 0))
    return fig, axs


def make_subplot_list(
    data: pd.DataFrame,
    var_rows,
    var_colors,
    var_x,
    var_y,
    axsize=(6, 3),
    x_scale="linear",
    y_scale="linear",
    cmap="viridis",
    use_baseline=False,
    share_x=False,
    smoothing=1.0,
):
    cmapper = ValueColorMapper(data[var_colors].unique(), cmap=cmap)
    nrows = data[var_rows].nunique()
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(axsize[0], nrows * axsize[1]),
        sharex=share_x,
    )
    axs[0].set_title(var_y)
    axs[0].set_xscale(x_scale)
    axs[-1].set_xlabel(var_x)
    for row_idx, (value_row, data_row) in enumerate(data.groupby(var_rows)):
        ax = axs[row_idx]
        ax.set_ylabel(value_row)
        ax.set_yscale(y_scale)
        for color_value, data_color in data_row.groupby(var_colors):
            ax.plot(
                data_color[var_x],
                data_color[var_y].ewm(alpha=smoothing).mean(),
                label=color_value,
                c=cmapper(color_value),
            )
            if "upper_bound" in data_color:
                ax.plot(
                    data_color[var_x],
                    data_color["upper_bound"],
                    c=cmapper(color_value),
                    linestyle="--",
                )
    ax.legend(loc=(1.04, 0), title=var_colors)
    fig.tight_layout()
    return fig, axs


def make_subplot_grid(
    data: pd.DataFrame,
    var_rows,
    var_cols,
    var_colors,
    var_x,
    var_y,
    axsize=(5, 3),
    x_scale="linear",
    y_scale="linear",
    share_x="col",
    share_y="row",
    cmap="viridis",
):
    cmapper = ValueColorMapper(data[var_colors].unique(), cmap=cmap)
    nrows = data[var_rows].nunique()
    ncols = data[var_cols].nunique()
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * axsize[0], nrows * axsize[1]),
        sharex=share_x,
        sharey=share_y,
    )
    for row_idx, (value_row, data_row) in enumerate(data.groupby(var_rows)):
        axs[row_idx, 0].set_ylabel(value_row)
        axs[row_idx, 0].set_yscale(y_scale)
        for col_idx, (value_col, data_col) in enumerate(data_row.groupby(var_cols)):
            ax = axs[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(value_col)
                ax.set_xscale(x_scale)
            elif row_idx == nrows - 1:
                ax.set_xlabel(var_x)
            for color_value, data_color in data_col.groupby(var_colors):
                if "upper_bound" in data_color:
                    ax.plot(
                        data_color[var_x],
                        data_color["upper_bound"],
                        c=cmapper(color_value),
                        linestyle="dashed",
                    )
                if "lower_bound" in data_color:
                    ax.plot(
                        data_color[var_x],
                        data_color["lower_bound"],
                        c=cmapper(color_value),
                        linestyle="dotted",
                    )
                ax.plot(
                    data_color[var_x],
                    data_color[var_y],
                    label=color_value,
                    c=cmapper(color_value),
                )
    ax.legend(loc=(1.04, 0), title=var_colors)
    # fig.suptitle(var_cols)
    fig.tight_layout()
    return fig, axs


def filter_params(data, params):
    result = []
    for param_i in params:
        data_tmp = data
        for key, value in param_i.items():
            data_tmp = data_tmp[data_tmp[key] == value]
        result.append(data_tmp)

    return pd.concat(result)


def get_best_within_std(
    data: pd.DataFrame,
    metrics,
    params_fixed=[],
    suffixes=["mean", "std"],
    modes="max",
):
    """
    Function generating a dictionary that contains boolean maps indicating which of the respective columns values are the best within their group of fixed parameter values.
    """
    if isinstance(modes, str):
        modes = [modes] * len(metrics)
    data_tmp = data.copy()

    best = {}
    for idx, metric in enumerate(metrics):
        # Get names of columns where mean values and standard deviations are stored
        col_mean = add_index_suffix(metric, suffixes[0])
        col_std = add_index_suffix(metric, suffixes[1])

        # Negate values in case smaller is better for the current metric
        if modes[idx] != "max":
            data_tmp[col_mean] *= -1

        # Calculate the best optimality bound given by `expected_value - std_dev`
        data_tmp["bound"] = data_tmp[col_mean] - data_tmp[col_std]
        if len(params_fixed) > 0:
            thresholds = data_tmp.groupby(params_fixed)["bound"].max().reset_index()
            data_tmp = data_tmp.merge(
                thresholds, on=params_fixed, suffixes=["", "_best"]
            )
        else:
            thresholds = data_tmp["bound"].max()
            data_tmp["bound_best"] = thresholds

        best[metric] = data_tmp[col_mean] > data_tmp["bound_best"]

        if isinstance(metric, tuple):
            data_tmp = data_tmp.drop(columns=["bound", "bound_best"], level=0)
        else:
            data_tmp = data_tmp.drop(columns=["bound", "bound_best"])

    return best


def merge_mean_std(
    data, metrics, suffixes=["mean", "std"], formats="{:.4f}", drop=True
):
    data_tmp = data.copy()
    if isinstance(formats, str):
        formats = [formats] * len(metrics)
    for fmt, metric in zip(formats, metrics):
        col_mean = add_index_suffix(metric, suffixes[0])
        col_std = add_index_suffix(metric, suffixes[1])

        data_tmp[metric] = data_tmp.apply(
            lambda row: fmt.format(row[col_mean]).lstrip("0")
            + "\u00B1"
            + fmt.format(row[col_std]).lstrip("0"),
            axis=1,
        )
        if drop:
            data_tmp = data_tmp.drop(columns=[col_mean, col_std])
    return data_tmp


def add_index_suffix(metric, suffix):
    if isinstance(metric, tuple):
        return metric[:-1] + (metric[-1] + "_" + suffix,)
    else:
        return metric + "_" + suffix


def apply_style_attribute(style, attribute="bfseries: ;", mask=None):
    attr_map = {key: np.where(mask[key], attribute, None) for key in mask.keys()}
    style.apply(lambda x: attr_map[x.name], subset=list(mask.keys()), axis=0)
    return style


def merge_and_bold_best(
    data,
    metrics,
    params_fixed,
    suffixes=["mean", "std"],
    modes="max",
    metric_formats="{:.4f}",
):
    data_tmp = data.copy()
    mask_best = get_best_within_std(
        data_tmp,
        metrics=metrics,
        params_fixed=params_fixed,
        suffixes=suffixes,
        modes=modes,
    )

    data_tmp = merge_mean_std(
        data_tmp, metrics=metrics, suffixes=suffixes, formats=metric_formats
    )
    return apply_style_attribute(data_tmp.style, mask=mask_best)


def col_to_header(data, new_header, index):
    data_tmp = data.copy()
    order_new_header = data_tmp[new_header].unique()
    order_old_header = data_tmp.columns.drop([new_header, *index])
    data_tmp = data_tmp.pivot(columns=[new_header], index=index)
    data_tmp = data_tmp.reorder_levels([1, 0], axis=1)
    data_tmp = data_tmp.reindex(
        pd.MultiIndex.from_product([order_new_header, order_old_header]), axis="columns"
    )
    return data_tmp


def get_topk_results(
    data,
    top_variable,
    other_variables,
    k=1,
    metric_to_show="accuracy",
):
    """ """
    data_mean = agg_dataframe(data, [*other_variables, top_variable], agg_fns=["mean"])

    idcs_topk = (
        data_mean.groupby([*other_variables])[f"{metric_to_show}_mean"]
        .nlargest(k)
        .index
    )
    lrs_topk = data_mean.loc[idcs_topk.get_level_values(-1)].set_index(
        [*other_variables]
    )[top_variable]

    data_topk = pd.merge(lrs_topk.reset_index(), data, how="inner")
    mean_topk = agg_dataframe(data_topk, cols_to_keep=[*other_variables])
    return mean_topk





def get_scheduler_params(df, column="schedule", suffix=""):
    for param in ["gamma", "factor", "maxlr"]:
        df[param] = df[column].apply(
            lambda x: float(x.split("=")[-1]) if param in x else np.NaN
        )
    df[column] = df[column].apply(lambda x: x.split("_")[0] + suffix)
    return df


def label_resets(row):
    if row["drift_confidence"] > 0:
        return row["schedule"] + " reset"
    else:
        return row["schedule"]
