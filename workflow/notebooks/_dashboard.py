# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot the dashboard for system-defining events."""

from typing import Dict, List, Optional

import pandas as pd
import pypsa
import xarray as xr
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.dates as mdates
import seaborn as sns

import geopandas as gpd
import cartopy.crs as ccrs

from _notebook_utilities import *
from _plot_affected_areas import *

import logging

# Suppress warnings and info messages from 'pypsa.io'
logging.getLogger("pypsa.io").setLevel(logging.ERROR)

cm = 1 / 2.54  # centimeters in inches


def collect_annual_values(
    ranked_years: pd.DataFrame,
    annual_cfs: pd.DataFrame,
    winter_load: pd.DataFrame,
    total_costs: Dict[int, pd.Series],
    flex_caps: pd.DataFrame,
    years: List[int],
) -> pd.DataFrame:
    """
    Collect annual values for dashboard display.

    Parameters:
    -----------
    ranked_years: pd.DataFrame
        Ranks of years by system costs, design year, and operational year
    annual_cfs: pd.DataFrame
        Solar and wind capacity factors per year
    winter_load: pd.DataFrame
        Winter load data
    total_costs: Dict[int, pd.Series]
        Total costs of each year
    flex_caps: pd.DataFrame
        Capacity of battery dischargers and fuel cells
    years: List[int]
        Years to process

    Returns:
    --------
    pd.DataFrame
        DataFrame containing collected annual values
    """
    annual_values = pd.DataFrame(index=years).astype(float)

    annual_values["ranked_costs"] = ranked_years["System costs"]
    annual_values["design_year"] = ranked_years["Prevents peaks"]
    annual_values["op_year"] = ranked_years["Causes peaks"]

    annual_values["solar_cf"] = annual_cfs["solar"].round(2)
    annual_values["wind_cf"] = annual_cfs["wind"].round(2)
    annual_values["winter_load"] = (
        winter_load["load"] / winter_load["load"].max()
    ).round(2)

    annual_values["total_costs"] = [total_costs[y].sum() for y in years]

    annual_values["battery_discharger"] = flex_caps["battery discharger"]
    annual_values["fuel_cells"] = flex_caps["H2 fuel cell"]
    return annual_values


def load_hull_data(
    config_name: str, periods: pd.DataFrame, techs: List[str], thres: List[float]
) -> Dict[str, List[ConvexHull]]:
    """
    Load hull data for different technologies and thresholds.

    Parameters:
    -----------
    config_name: str
        Configuration name
    periods: pd.DataFrame
        DataFrame with periods information
    techs: List[str]
        List of technologies to load hulls for
    thres: List[float]
        List of thresholds corresponding to each technology

    Returns:
    --------
    Dict[str, List[ConvexHull]]
        Dictionary mapping technologies to lists of convex hulls
    """
    """Load the hull data for the different technologies and thresholds."""
    hulls = {tech: [] for tech in techs}
    for tech, t in zip(techs, thres):
        hulls[tech].extend(
            [
                ConvexHull(
                    pd.read_csv(
                        f"processing_data/{config_name}/maps/{tech}/hull_{t}_event{i}.csv",
                        index_col=0,
                    )
                )
                for i in range(len(periods))
            ]
        )
    return hulls


def plot_dashboard(
    config_name: str,
    event_nr: int,
    stats_periods: pd.DataFrame,
    annual_values: pd.DataFrame,
    kpis: List[str],
    kpi_names: List[str],
    hulls_coll: Dict[str, List[ConvexHull]],
    hulls_markers_names: List[str],
    fc_flex: pd.DataFrame,
    onshore_regions: gpd.GeoDataFrame,
    n: pypsa.Network,
    projection: ccrs.Projection,
    categories: List[str],
    cat_names: List[str],
    label_names: Dict[str, List[str]],
    gen_stacks: pd.DataFrame,
    time_window: pd.Timedelta,
    total_load: pd.DataFrame,
    freq: str,
    save: bool = False,
) -> plt.Figure:
    """
    Plot a dashboard with KPIs, map, annual stats, and generation stack plot.

    Parameters:
    -----------
    config_name: str
        Name of the configuration
    event_nr: int
        Event number to display
    stats_periods: pd.DataFrame
        Statistics of all periods
    annual_values: pd.DataFrame
        Annual values data
    kpis: List[str]
        Key performance indicators to display
    kpi_names: List[str]
        Names of the KPIs for display
    hulls_coll: Dict[str, List[ConvexHull]]
        Hull data for different technologies
    hulls_markers_names: List[str]
        Names of the hull markers
    fc_flex: pd.DataFrame
        Fuel cell flexibility data
    onshore_regions: gpd.GeoDataFrame
        Onshore regions data
    n: pypsa.Network
        PyPSA network object
    projection: ccrs.Projection
        Map projection
    categories: List[str]
        Categories for the annual stats
    cat_names: List[str]
        Names of the categories
    label_names: Dict[str, List[str]]
        Label names for the categories
    gen_stacks: pd.DataFrame
        Generation stack data
    time_window: pd.Timedelta
        Time window for the generation stack plot
    total_load: pd.DataFrame
        Total load data
    freq: str
        Frequency for resampling the data
    save: bool, default False
        Whether to save the plot

    Returns:
    --------
    plt.Figure
        The dashboard figure
    """
    fig, axd = plt.subplot_mosaic(
        mosaic=[["kpi", "map", "stats"], ["gen", "gen", "gen"]],
        width_ratios=[1, 1, 1.6],
        height_ratios=[2, 1],
        gridspec_kw={"hspace": 0.8},
        per_subplot_kw={
            "map": {"projection": projection},
        },
        figsize=(18 * cm, 14 * cm),
    )

    period = stats_periods.loc[event_nr]

    ### PLOT KPIs
    ax = axd["kpi"]
    # Generate subaxes for the stripplots of each KPI.
    sub_axes = []
    n_kpis = len(kpis)
    main_box = ax.get_position()
    sub_ax_height = main_box.height / n_kpis

    # Create horizontal subplots
    for (i, kpi), name in zip(enumerate(kpis), kpi_names):
        bottom = main_box.y0 + i * sub_ax_height
        # height = sub_ax_height
        sub_ax = fig.add_axes([main_box.x0, bottom, main_box.width, sub_ax_height])
        sub_axes.append(sub_ax)

        # First make stripplot of all SDEs
        sns.stripplot(
            data=stats_periods,  # Adjust duration as needed
            x=kpi,
            ax=sub_ax,
            jitter=0.1,
            alpha=0.2,
            size=3,
            color="grey",
            legend=False,
        )

        # Annotate the min and max value with their value
        minval, maxval = stats_periods[kpi].min(), stats_periods[kpi].max()
        for val, text in zip([minval, maxval], ["min", "max"]):
            if kpi in ["wind_anom", "avg_rel_load", "normed_price_std"]:
                sub_ax.text(
                    val,
                    -0.15,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="grey",
                )
            else:
                sub_ax.text(
                    val,
                    -0.15,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="grey",
                )

        # Plot all other values that are inside the cluster
        sns.stripplot(
            data=stats_periods[stats_periods.cluster == period.cluster],
            x=kpi,
            ax=sub_ax,
            jitter=0.1,
            alpha=0.4,
            size=3,
            marker="D",
            color="blue",
            legend=False,
        )

        # Plot the actual value
        sns.stripplot(
            data=stats_periods[stats_periods.index == period.name],
            x=kpi,
            ax=sub_ax,
            jitter=0,
            alpha=0.8,
            size=3,
            marker="D",
            color="red",
            legend=False,
        )
        # Annotate value of selected period.
        if kpi in ["wind_anom", "avg_rel_load", "normed_price_std"]:
            sub_ax.text(
                period[kpi],
                0.5,
                f"{period[kpi]:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
            )
        else:
            sub_ax.text(
                period[kpi],
                0.5,
                f"{period[kpi]:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
            )

        # Add the kpi as yticklabel and no x and y labels.
        sub_ax.text(
            0.5,
            sub_ax_height + 0.6,
            name,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            transform=sub_ax.transAxes,
        )
        sub_ax.set_xticklabels([])
        sub_ax.set_ylabel("")
        sub_ax.set_xlabel("")
        sub_ax.tick_params(length=0, axis="both")

        # Remove border around axes.
        sub_ax.axis("off")
    sub_axes[0].legend(
        ["All SDEs", "Same cluster", "SDE"],
        loc="upper left",
        bbox_to_anchor=(0.25, -0.2),
        fontsize=7,
        ncols=1,
    )
    axd["kpi"].set_visible(False)
    

    ### MAP
    ax = axd["map"]
    legend_elements = plot_affected_areas(
        config_name,
        period,
        event_nr,
        [hulls_coll[tech][event_nr] for tech in hulls_coll.keys()],
        list(hulls_coll.keys()),
        hulls_markers_names,
        ["#235ebc", "#dd2e23", "green", "#c251ae"],
        fc_flex,
        "fuel_cells",
        mpl.colors.Normalize(vmin=0, vmax=1),
        "Purples",
        onshore_regions,
        n,
        projection,
        ax=ax,
    )
    ax.legend(
        handles=legend_elements, bbox_to_anchor=(0.5, 0), loc="upper center", fontsize=7
    )

    ## ANNUAL STATS
    ax = axd["stats"]


    # Get annual values.
    net_year = get_net_year(pd.Timestamp(period.start))
    difficulty_ranks = [
        annual_values.loc[net_year, c]
        for c in ["ranked_costs", "design_year", "op_year"]
    ]
    weather_vals = [
        annual_values.loc[net_year, c] for c in ["solar_cf", "wind_cf", "winter_load"]
    ]
    min_weather_vals = [
        annual_values[c].min() for c in ["solar_cf", "wind_cf", "winter_load"]
    ]
    max_weather_vals = [
        annual_values[c].max() for c in ["solar_cf", "wind_cf", "winter_load"]
    ]

    # Share of total costs in period, recovered costs of hydrogen infrastructure
    costs_vals = [
        period[col]
        for col in [
            "share_total_costs",
            "recovered_h2_costs",
            "recovered_battery_costs",
        ]
    ]
    min_costs_vals = [
        stats_periods[col].min()
        for col in [
            "share_total_costs",
            "recovered_h2_costs",
            "recovered_battery_costs",
        ]
    ]
    max_costs_vals = [
        stats_periods[col].max()
        for col in [
            "share_total_costs",
            "recovered_h2_costs",
            "recovered_battery_costs",
        ]
    ]

    # Capacity of battery dischargers and fuel cells
    cap_vals = [
        annual_values.loc[net_year, c] for c in ["battery_discharger", "fuel_cells"]
    ]
    min_caps = [annual_values[c].min() for c in ["battery_discharger", "fuel_cells"]]
    max_caps = [annual_values[c].max() for c in ["battery_discharger", "fuel_cells"]]

    # Generate subaxes of ax where each category gets its own axes.
    sub_axes = []
    n_axes = len(categories)
    main_box = ax.get_position()
    sub_ax_width = main_box.width / n_axes

    for (i, cat), name in zip(enumerate(categories), cat_names):
        left = main_box.x0 + i * sub_ax_width
        width = sub_ax_width
        sub_ax = fig.add_axes([left, main_box.y0, width, main_box.height])
        sub_axes.append(sub_ax)

        if cat == "difficulty":
            sub_ax.bar(
                x=[x - 0.25 for x in range(len(difficulty_ranks))],
                height=[80 - r for r in difficulty_ranks],
                width=0.5,
                color="darkgrey",
                alpha=0.8,
            )
            sub_ax.set_ylim(0, 88)
            # Annotate value
            for i, val in enumerate(difficulty_ranks):
                sub_ax.text(
                    i - 0.25,
                    80 - val,
                    f"{val}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )

        if cat == "weather":
            sub_ax.bar(
                x=[x - 0.25 for x in range(len(weather_vals))],
                bottom=min_weather_vals,
                height=[a - m for a, m in zip(max_weather_vals, min_weather_vals)],
                width=0.5,
                color=[c for c in ["#f9d002", "#235ebc", "#dd2e23"]],
                alpha=0.8,
            )
            sub_ax.hlines(weather_vals[0], -0.5, 0, color="black", lw=0.5)
            sub_ax.hlines(weather_vals[1], 0.5, 1, color="black", lw=0.5)
            sub_ax.hlines(weather_vals[2], 1.5, 2, color="black", lw=0.5)
            sub_ax.set_ylim(0, 1.1)

            # Annotate min, avg, max values
            for (
                i,
                (avg, minval, maxval, xpos),
            ) in enumerate(
                zip(weather_vals, min_weather_vals, max_weather_vals, [0.8, -0.3, 2.8])
            ):
                sub_ax.text(
                    i - 0.2,
                    minval,
                    f"{minval:.2f}",
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="grey",
                )
                sub_ax.text(
                    xpos,
                    avg,
                    f"{avg:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )
                sub_ax.text(
                    i - 0.2,
                    maxval,
                    f"{maxval:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="grey",
                )
        if cat == "cost":
            sub_ax.bar(
                x=[x - 0.25 for x in range(len(costs_vals))],
                bottom=min_costs_vals,
                height=[a - m for a, m in zip(max_costs_vals, min_costs_vals)],
                width=0.5,
                color="green",
                alpha=0.8,
            )
            sub_ax.hlines(costs_vals[0], -0.5, 0, color="black", lw=0.5)
            sub_ax.hlines(costs_vals[1], 0.5, 1, color="black", lw=0.5)
            sub_ax.hlines(costs_vals[2], 1.5, 2, color="black", lw=0.5)
            sub_ax.set_ylim(0, 1.1)
            # Annotate min, avg, max values
            for i, (avg, minval, maxval, xpos) in enumerate(
                zip(costs_vals, min_costs_vals, max_costs_vals, [-1.2, 1.8, 2.8])
            ):
                sub_ax.text(
                    i - 0.2,
                    minval,
                    f"{minval:.2f}",
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="grey",
                )
                sub_ax.text(
                    xpos,
                    avg,
                    f"{avg:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )
                sub_ax.text(
                    i - 0.2,
                    maxval,
                    f"{maxval:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="grey",
                )
        if cat == "other":
            sub_ax.bar(
                x=[x - 0.25 for x in range(len(cap_vals))],
                bottom=min_caps,
                height=[a - m for a, m in zip(max_caps, min_caps)],
                width=0.5,
                color="purple",
                alpha=0.8,
            )
            sub_ax.hlines(cap_vals[0], -0.5, 0, color="black", lw=0.5)
            sub_ax.hlines(cap_vals[1], 0.5, 1, color="black", lw=0.5)
            sub_ax.set_ylim(0, 250)
            # Annotate min, avg, max values
            for i, (avg, minval, maxval, xpos) in enumerate(
                zip(cap_vals, min_caps, max_caps, [0.5, 1.5])
            ):
                sub_ax.text(
                    i - 0.2,
                    minval,
                    f"{minval:.0f}",
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="grey",
                )
                sub_ax.text(
                    xpos,
                    avg,
                    f"{avg:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )
                sub_ax.text(
                    i - 0.2,
                    maxval,
                    f"{maxval:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="grey",
                )

        # Add the value as yticklabel and no x and y labels.
        sub_ax.text(
            0.25,
            1,
            name,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            transform=sub_ax.transAxes,
        )
        # Set fixed locator
        sub_ax.set_xticks(range(len(label_names[cat])))
        sub_ax.set_xticklabels(label_names[cat], rotation=90, fontsize=7, ha="center")

        sub_ax.set_yticklabels([])
        sub_ax.set_ylabel("")
        sub_ax.set_xlabel("")
        sub_ax.tick_params(length=0, axis="both")
        sub_ax.set_xlim(-0.5, len(label_names[cat]) + 0.5)

        # Remove border around axes.
        for loc in ["top", "right", "left", "right"]:
            sub_ax.spines[loc].set_visible(False)
        # sub_ax.axis("off")

    axd["stats"].set_visible(False)
    ax.text(0, 0.95, "(c)", fontsize=8, transform=ax.transAxes)

    ## GENERATION STACK PLOT
    ax = axd["gen"]
    plot_gen_stack(
        gen_stacks,
        total_load,
        pd.Timestamp(stats_periods.loc[event_nr, "start"]) - time_window,
        pd.Timestamp(stats_periods.loc[event_nr, "end"]) + time_window,
        stats_periods,
        freq=freq,
        ax=ax,
    )
    ax.hlines(y=0, xmin=gen_stacks.index[0], xmax=gen_stacks.index[-1], color="black", lw=0.5)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(labelsize=7, length=0, which="both", axis="both")
    ax.set_title("Generation stack [GW]", fontsize=8)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(axis="x", which="both", color="grey", linestyle=":", linewidth=0.5)
    ax.grid(axis="y", which="major", color="grey", linestyle="--", linewidth=0.5)
    ax.grid(axis="y", which="minor", color="grey", linestyle=":", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    pretty_labels = ["Biomass", "Nuclear", "Run-of-river", "Battery discharge", "Battery charge", "Hydro", "PHS", "Fuel cells", "Electrolysis", "Solar", "Offshore wind", "Onshore wind", "SDE", "Load"]
    

    ax.legend(labels=pretty_labels, handles = handles, bbox_to_anchor=(0.95, -0.18), fontsize=7, ncols=5)

    # Add labels to plots.
    for numb, coord in zip(["(a)", "(b)", "(c)", "(d)"], [(0.09, 0.88), (0.33, 0.88), (0.57, 0.88), (0.09, 0.3)]):
        fig.text(
            coord[0],
            coord[1],
            numb,
            fontsize=8,
        )

    if save:
        plt.savefig(
            f"./plots/{config_name}/dashboard/event_{event_nr}.pdf", bbox_inches="tight"
        )


def plot_gen_stack(
    gen_df: pd.DataFrame,
    total_load: pd.DataFrame,
    start,
    end,
    periods: pd.DataFrame,
    freq: str = "3H",
    ax: Optional = None,
):
    """Plot the generation stack with highlighted difficult periods."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    p = gen_df.loc[start:end].resample(freq).mean() / 1e3
    # Ensure we have no leap days
    p = p[~((p.index.month == 2) & (p.index.day == 29))]
    p_neg = p.clip(upper=0)
    p = p.clip(lower=0)

    loads = total_load.loc[start:end, "0"].resample(freq).mean() / 1e3
    # Ensure we have no leap days
    loads = loads[~((loads.index.month == 2) & (loads.index.day == 29))]

    gen_colors = [
        "#baa741",
        "#ff8c00",
        "#3dbfb0",
        "#5d4e29",
        "#88a75b",
        "#298c81",
        "#51dbcc",
        "#c251ae",
        "#ff29d9",
        "#f9d002",
        "#6895dd",
        "#235ebc",
    ]

    # Plot the generation stack.
    ax.stackplot(p.index, p.transpose(), colors=gen_colors, labels=p.columns)
    ax.stackplot(p.index, p_neg.transpose(), colors=gen_colors)

    # Plot the difficult periods.
    ymin, ymax = ax.get_ylim()
    for i, period in periods.iterrows():
        if (
            pd.to_datetime(periods.loc[i, "start"]) > start
            and pd.to_datetime(periods.loc[i, "end"]) < end
        ):
            ax.fill_between(
                [pd.Timestamp(period.start), pd.Timestamp(period.end)],
                ymin,
                ymax,
                color="grey",
                alpha=0.3,
                label="SDE",
            )

    # Plot load.
    ax.plot(loads, ls="--", color="black", label="load", lw=1)
    ax.set_xlim(start, end)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 2), ncols=4, fontsize=7)
    ax.set_ylabel("GW")
    plt.tight_layout()



if __name__ == "__main__":
    config_name = "stressful-weather"
    config, scenario_def, years, opt_networks = load_opt_networks(
        config_name, load_networks=False
    )
    periods = load_periods(config)
    projection = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    cluster_nr = 4

    # Load onshore and offshore regions for shapefile.
    onshore_regions = gpd.read_file(
        f"../pypsa-eur/resources/{config_name}/weather_year_1941/regions_onshore_base_s_90.geojson"
    )
    # Load one network for reference and the layout.
    n = pypsa.Network(
        "../pypsa-eur/results/stressful-weather/weather_year_1941/networks/base_s_90_elec_lc1.25_Co2L.nc"
    )

    # Load all data we might need that is pre-generated in `generate_data_for_analysis.py`.
    folder = f"./processing_data/{config_name}"
    # Load: total load, winter load
    total_load = pd.read_csv(f"{folder}/total_load.csv", index_col=0, parse_dates=True)
    winter_load = pd.read_csv(f"{folder}/winter_load.csv", index_col=0)
    # Annual CFS for solar and wind
    annual_cfs = pd.read_csv(f"{folder}/annual_cfs.csv", index_col=0)
    # Costs: total electricity costs
    total_costs_df = pd.read_csv(f"{folder}/total_costs.csv", index_col=[0, 1])
    ## SDEs
    stats_periods = pd.read_csv(f"{folder}/stats_periods.csv", index_col=0)
    gen_stacks = pd.read_csv(f"{folder}/gen_stacks.csv", index_col=0, parse_dates=True)
    # Flexibility usage and capacities
    nodal_flex_p = pd.read_csv(f"{folder}/nodal_flex_p.csv", index_col=[0, 1])
    nodal_flex_u = xr.open_dataset(f"processing_data/{config_name}/nodal_flex_u.nc")
    fc_flex = nodal_flex_u["H2 fuel cell"].to_pandas().T
    fc_flex.index = total_load.index
    system_flex_p = (nodal_flex_p.unstack().sum(axis="rows") / 1e3).round(1)
    flex_caps = system_flex_p[["battery discharger", "H2 fuel cell"]].unstack(level=0)
    # Ranked years: NOTE that this refers to highest load shedding, not over the annual sum
    ranked_years = pd.read_csv(f"processing_data/stressful-weather-sensitivities/ranked_years.csv", index_col=0)
    total_costs = {}
    for year in years:
        df = total_costs_df.loc[year]
        df.index = pd.to_datetime(df.index)
        total_costs[year] = df["0"]

    # Generate a dataframe with all necessary annual values:
    annual_values = collect_annual_values(
        ranked_years, annual_cfs, winter_load, total_costs, flex_caps, years
    )

    # Load clusters
    clusters = pd.read_csv(
        f"clustering/{config_name}/clustered_vals_{cluster_nr}.csv", index_col=0
    )["cluster"]
    stats_periods["cluster"] = clusters

    # Load hulls
    hulls_coll = load_hull_data(
        config_name,
        periods,
        techs=["wind_anom", "load_anom", "price"],
        thres=[0.75, 0.9, 0.9],
    )
    hulls_markers_names = [
        "Wind anomaly (-)",
        "Load anomaly (+)",
        "Shadow price (+)",
        "Fuel cell discharge (+)",
    ]

    # Plot dashboard.
    for event_nr in periods.index:
        plot_dashboard(
            config_name="stressful-weather",
            event_nr=event_nr,
            stats_periods=stats_periods,
            annual_values=annual_values,
            kpis=[
                "highest_net_load",
                "avg_net_load",
                "wind_anom",
                "avg_rel_load",
                "max_fc_discharge",
                "duration",
                "normed_price_std",
            ],
            kpi_names=[
                "Peak net load [GW]",
                "Avg. net load [GW]",
                "Wind CF anomaly",
                "Avg. rel. load",
                "Max. Fuel cell discharge [GW]",
                "Duration [h]",
                "Regional price imbalance",
            ],
            hulls_coll=hulls_coll,
            hulls_markers_names=hulls_markers_names,
            fc_flex=fc_flex,
            onshore_regions=onshore_regions,
            n=n,
            projection=projection,
            categories=["difficulty", "weather", "cost", "other"],
            cat_names=["Difficulty rank", "Weather", "Costs of SDE", "Flexibility"],
            label_names={
                "difficulty": ["Total system costs", "LS design", "LS operation"],
                "weather": ["Annual solar CF", "Annual wind CF", "Winter load"],
                "cost": [
                    "Share of total costs",
                    "Recovery of FC inv.",
                    "Recovery of battery inv.",
                ],
                "other": [
                    "Battery discharger \n capacity[GW]",
                    "Fuel cell capacity \n [GW]",
                ],
            },
            gen_stacks=gen_stacks,
            time_window=pd.Timedelta("7d"),
            total_load=total_load,
            freq="3H",
            save=True,
        )
