# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Generate data for the notebooks in which we analyse and plot the data.
"""

import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from _notebook_utilities import *


def average_across_years(df, years):
    """
    Calculate the average of data across multiple years.

    This function takes a DataFrame containing time series data and a list of years,
    and returns a DataFrame with the average values for each hour across the specified years.

    Parameters:
    -----------
    df: pd.DataFrame
        The input DataFrame containing time series data.
    years: List[int]
        A list of years to be considered for averaging.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the average values for each hour across the specified years.
        The index of the returned DataFrame is a date range from July 1st of the first year
        to June 30th of the following year, with hourly frequency.
    """
    df_help = df.copy()
    df_help.index = range(len(df_help))
    list_help = [
        df_help.loc[i * 8760 : (i + 1) * 8760 - 1] for i in range(len(df_help) // 8760)
    ]
    for li in list_help:
        li.index = range(8760)
    avg_df = pd.concat(list_help).groupby(level=0).mean()
    avg_df.index = pd.date_range(
        f"{years[0]}-07-01", f"{years[0] + 1}-06-30 23:00", freq="h"
    )
    return avg_df


def compute_hydro_phs_costs(
    opt_networks: Dict[int, pypsa.Network],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the costs associated with hydro and pumped hydro storage (PHS) units
    from a given set of optimized networks.

    Parameters:
    -----------
    opt_networks: Dict[int, pypsa.Network]
        A dictionary of optimized network objects. Each network
        object should have storage units with associated time
        series data for power output and energy balance multipliers.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two pandas DataFrames:
        - hydro_costs: Costs associated with hydro storage units.
        - phs_costs: Costs associated with pumped hydro storage (PHS) units.
    """
    hydro_costs = []
    phs_costs = []
    for n in opt_networks.values():
        hydro_i = n.storage_units.index[n.storage_units.carrier.isin(["hydro"])]
        phs_i = n.storage_units.index[n.storage_units.carrier.isin(["PHS"])]
        hydro_costs.append(
            n.storage_units_t.p.loc[:, hydro_i]
            * n.storage_units_t.mu_energy_balance.loc[:, hydro_i]
        )
        phs_costs.append(
            n.storage_units_t.p.loc[:, phs_i]
            * n.storage_units_t.mu_energy_balance.loc[:, phs_i]
        )
    hydro_costs = pd.concat(hydro_costs, axis=0)
    phs_costs = pd.concat(phs_costs, axis=0)
    return hydro_costs, phs_costs


def compute_longest_deficit(years: List[int], net_load: pd.Series) -> pd.DataFrame:
    """
    Computes the longest deficit period for each year in the given range.

    Parameters:
    -----------
    years: List[int]
        List of years to analyze.
    net_load: pd.Series
        Series containing net load data with a datetime index.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ["start", "end", "hours", "mean", "total"] for each year.
        - "start": The start time of the longest deficit period.
        - "end": The end time of the longest deficit period.
        - "hours": The duration of the longest deficit period in hours.
        - "mean": The mean net load during the longest deficit period.
        - "total": The total net load during the longest deficit period.
    """
    longest_deficit = pd.DataFrame(
        index=years, columns=["start", "end", "hours", "mean", "total"]
    )
    for y in years:
        helper_df = pd.DataFrame(net_load.loc[f"{y}-07-01" : f"{y + 1}-06-30"])
        helper_df["deficit"] = 0
        helper_df.loc[helper_df.index[0], "deficit"] = 0
        for i in range(1, len(helper_df.index)):
            if net_load.loc[helper_df.index[i]] > 0:
                helper_df.loc[helper_df.index[i], "deficit"] = (
                    helper_df.loc[helper_df.index[i - 1], "deficit"] + 1
                )
            else:
                helper_df.loc[helper_df.index[i], "deficit"] = 0
        longest_deficit.loc[y, "end"] = helper_df["deficit"].idxmax()
        longest_deficit.loc[y, "start"] = helper_df["deficit"].idxmax() - pd.Timedelta(
            hours=helper_df["deficit"].max()
        )
        longest_deficit.loc[y, "hours"] = helper_df["deficit"].max()
        longest_deficit.loc[y, "mean"] = (
            helper_df.loc[
                longest_deficit.loc[y, "start"] : longest_deficit.loc[y, "end"]
            ]["Net load"]
            .mean()
            .round(0)
        )
        longest_deficit.loc[y, "total"] = (
            helper_df.loc[
                longest_deficit.loc[y, "start"] : longest_deficit.loc[y, "end"]
            ]["Net load"]
            .sum()
            .round(0)
        )
    return longest_deficit


def compute_system_anomaly(
    periods,
    years,
    total_load,
    wind_cf,
    solar_cf,
    avg_load,
    avg_wind,
    avg_solar,
    opt_networks,
):
    """
    Compute system anomalies for given periods.

    Parameters:
    periods (pd.DataFrame): DataFrame containing the periods with 'start' and 'end' columns.
    years (list): List of years corresponding to the periods.
    total_load (pd.DataFrame): DataFrame containing the total load data.
    wind_cf (pd.DataFrame): DataFrame containing the wind capacity factors.
    solar_cf (pd.DataFrame): DataFrame containing the solar capacity factors.
    avg_load (pd.DataFrame): DataFrame containing the average load data.
    avg_wind (pd.DataFrame): DataFrame containing the average wind data.
    avg_solar (pd.DataFrame): DataFrame containing the average solar data.
    opt_networks (dict): Dictionary containing the optimized networks for each year.

    Returns:
    pd.DataFrame: DataFrame containing the system anomalies with columns:
                    ["Net load anomaly", "Load anomaly", "Wind anomaly", "Solar anomaly"].
    """
    all_system_anomaly = []
    for i, row in periods.iterrows():
        net_year = get_year_period(row)
        start_year = years[0] if row["start"].month >= 7 else years[0] + 1
        end_year = years[0] if row["end"].month >= 7 else years[0] + 1

        shifted_start = pd.Timestamp(f"{start_year}-{str(row['start'])[4:]}")
        shifted_end = pd.Timestamp(f"{end_year}{str(row['end'])[4:]}")

        n = opt_networks[net_year]
        wind_i = n.generators.index[
            n.generators.carrier.isin(
                ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]
            )
        ]
        solar_i = n.generators.index[n.generators.carrier.isin(["solar", "solar-hsat"])]
        wind_p = n.generators.p_nom_opt.loc[wind_i]
        solar_p = n.generators.p_nom_opt.loc[solar_i]

        load_anomaly = (
            total_load.loc[row["start"] : row["end"]]
            - (avg_load.loc[shifted_start:shifted_end].sum(axis="columns")).values
        )
        wind_anomaly = (wind_cf.loc[row["start"] : row["end"]] @ wind_p) - (
            avg_wind.loc[shifted_start:shifted_end] @ wind_p
        ).values
        solar_anomaly = (solar_cf.loc[row["start"] : row["end"]] @ solar_p) - (
            avg_solar.loc[shifted_start:shifted_end] @ solar_p
        ).values

        net_load_anomaly = (
            load_anomaly - wind_anomaly - solar_anomaly
        )  # note the signs as net load is defined as load - wind_prod - solar_prod
        system_anomaly = pd.concat(
            [net_load_anomaly, load_anomaly, wind_anomaly, solar_anomaly],
            axis="columns",
        )
        system_anomaly.columns = [
            "Net load anomaly",
            "Load anomaly",
            "Wind anomaly",
            "Solar anomaly",
        ]
        all_system_anomaly.append(system_anomaly)
    all_system_anomaly = pd.concat(all_system_anomaly, axis="index").round(0)
    return all_system_anomaly


def compute_used_flex(periods, nodal_flex_p, df, disp_tech):
    """
    Compute the used flexibility for each node over specified periods.

    Parameters:
    periods (pd.DataFrame): DataFrame containing the periods with 'start' and 'end' columns.
    nodal_flex_p (pd.DataFrame): DataFrame with nodal flexibility percentages, indexed by node and year.
    df (pd.DataFrame): DataFrame containing dispatchable technology data.
    disp_tech (list): List of dispatchable technologies.

    Returns:
    pd.DataFrame: DataFrame containing the used flexibility for each dispatchable technology over the specified periods.
    """
    all_used_flexibility = []
    used_flex = {}
    nodes = list(nodal_flex_p.index.levels[0])
    for i, row in periods.iterrows():
        net_year = get_year_period(row)
        for node in nodes:
            ds = pd.DataFrame(columns=disp_tech)
            ds = (
                df[disp_tech].loc[node] * nodal_flex_p[disp_tech].loc[(node, net_year)]
            ).loc[row["start"] : row["end"]]
            used_flex[node] = ds
        used_flexibility = pd.DataFrame(columns=disp_tech, index=ds.index)
        for t in disp_tech:
            used_flexibility.loc[:, t] = (
                pd.concat([used_flex[node][t] for node in nodes], axis="columns")
                .sum(axis="columns")
                .round(0)
            )
        all_used_flexibility.append(used_flexibility)
    all_used_flexibility = pd.concat(all_used_flexibility, axis="index")
    return all_used_flexibility


def compute_net_load(opt_networks, n):
    """
    Parameters:
    opt_networks (dict): A dictionary where keys are years and values are network objects containing load and generation data.
    n (Network): A network object containing generator information.
    Returns:
    pd.Series: A pandas Series representing the net load (total load minus solar and wind generation) for each year concatenated together.
    """
    load = pd.concat(
        [opt_networks[y].loads_t.p.sum(axis="columns") for y in opt_networks.keys()],
        axis=0,
    )
    solar_i = n.generators.loc[n.generators.carrier.isin(["solar", "solar-hsat"])].index
    wind_i = n.generators.loc[n.generators.carrier.str.contains("wind")].index

    solar_gen = pd.concat(
        [
            opt_networks[y].generators_t.p[solar_i].sum(axis="columns")
            for y in opt_networks.keys()
        ],
        axis=0,
    )
    wind_gen = pd.concat(
        [
            opt_networks[y].generators_t.p[wind_i].sum(axis="columns")
            for y in opt_networks.keys()
        ],
        axis=0,
    )

    net_load = load - solar_gen - wind_gen
    net_load = net_load.rename("Net load")

    return net_load


def compute_flex_periods_anomalies(flex, periods, years):
    """
    Compute anomalies in system flexibility during specific periods and peak hours.

    Parameters:
    flex (pd.DataFrame): DataFrame containing system flexibility data indexed by time.
    periods (pd.DataFrame): DataFrame containing periods of interest with 'start', 'end', and 'peak_hour' columns.
    years (list): List of years to consider for averaging and anomaly calculations.

    Returns:
    tuple: A tuple containing:
        - avg_flex (pd.DataFrame): Average system flexibility across the specified years.
        - periods_flex (pd.DataFrame): System flexibility during the specified periods.
        - periods_anomaly_flex (pd.DataFrame): Anomalies in system flexibility during the specified periods.
        - peak_flex (pd.DataFrame): System flexibility during peak hours of the specified periods.
        - peak_anomaly_flex (pd.DataFrame): Anomalies in system flexibility during peak hours of the specified periods.
    """
    # Average system flexibility
    avg_flex = average_across_years(flex, years)
    # During system-defining events
    periods_flex = pd.concat(
        flex.loc[row["start"] : row["end"]] for i, row in periods.iterrows()
    )
    # Anomalies
    periods_anomaly_flex = anomaly_index_shift(periods_flex, avg_flex, years)
    # Peak hour and flex
    peak_flex = pd.concat(
        [flex.loc[periods.loc[i, "peak_hour"]] for i in periods.index],
        axis=1,
    ).T
    peak_anomaly_flex = anomaly_index_shift(peak_flex, avg_flex, years)
    return avg_flex, periods_flex, periods_anomaly_flex, peak_flex, peak_anomaly_flex


def characteristics_sdes(
    opt_networks: Dict[int, pypsa.Network],
    periods: pd.DataFrame,
    all_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the characteristics of the system-defining events.

    Parameters:
    -----------
    opt_networks: Dict[int, pypsa.Network]
        Dictionary of optimized PyPSA networks
    periods: pd.DataFrame
        DataFrame with start and end times of system-defining events
    all_prices: pd.DataFrame
        DataFrame with electricity prices

    Returns:
    --------
    pd.DataFrame
        DataFrame with characteristics of system-defining events
    """
    stores_periods = periods.copy()

    # Discharge behaviour in periods.
    for i, row in periods.iterrows():
        n = opt_networks[get_year_period(row)]
        soc_start = (
            n.stores_t["e"].filter(like="H2", axis="columns").loc[row["start"]].sum()
        )
        soc_end = (
            n.stores_t["e"].filter(like="H2", axis="columns").loc[row["end"]].sum()
        )
        stores_periods.loc[i, "discharge"] = soc_start - soc_end
        stores_periods.loc[i, "relative_discharge"] = (
            100
            * stores_periods.loc[i, "discharge"]
            / (n.stores.loc[n.stores.carrier == "H2", "e_nom_opt"].sum())
        ).round(1)

        # Count the number of empty H2 storage units at the end of the event.
        su_i = n.stores.loc[n.stores.carrier == "H2"].index
        su_i = n.stores.loc[su_i][n.stores.e_nom_opt > 1].index
        stores_periods.loc[i, "empty"] = (
            (
                n.stores_t["e"].loc[row["end"], su_i]
                < 0.02 * (n.stores.loc[su_i, "e_nom_opt"])
            ).sum()
            / len(su_i)
            * 100
        )

        fc_i = n.links.loc[n.links.carrier == "H2 fuel cell"].index
        fc_i = n.links.loc[fc_i][n.links.p_nom_opt > 0.3].index

        # Compute the maximal discharge from fuel cell.
        stores_periods.loc[i, "max_fc_discharge"] = (
            n.links_t.p1[fc_i]
            .loc[row["start"] : row["end"]]
            .abs()
            .sum(axis="columns")
            .max()
            / 1e3
        )

        # Count the number of H2 storage units who reached their power capacity during the event.
        p_lims = n.links.loc[fc_i, "p_nom_opt"] * n.links.loc[fc_i, "efficiency"]
        stores_periods.loc[i, "affected_fc_p_lim"] = (
            100
            * (
                (
                    n.links_t.p1[fc_i].loc[row["start"] : row["end"]].abs()
                    > 0.99 * p_lims
                ).sum()
                > 0
            ).sum()
            / len(fc_i)
        )
        stores_periods.loc[i, "ratio_p_lim"] = (
            100
            * (
                (
                    (
                        n.links_t.p1[fc_i].loc[row["start"] : row["end"]].abs()
                        > 0.99 * p_lims
                    ).sum()
                )
                / len(n.links_t.p1.loc[row["start"] : row["end"], fc_i])
            ).mean()
        )

        # Do the same for the two peek period before the start of the event.
        stores_periods.loc[i, "2w_prior_affected_fc_p_lim"] = (
            100
            * (
                (
                    n.links_t.p1[fc_i]
                    .loc[row["start"] - dt.timedelta(days=14) : row["start"]]
                    .abs()
                    > 0.99 * p_lims
                ).sum()
                > 0
            ).sum()
            / len(fc_i)
        )
        stores_periods.loc[i, "2w_prior_ratio_p_lim"] = (
            100
            * (
                (
                    (
                        n.links_t.p1[fc_i]
                        .loc[row["start"] - dt.timedelta(days=14) : row["start"]]
                        .abs()
                        > 0.99 * p_lims
                    ).sum()
                )
                / len(
                    n.links_t.p1[fc_i].loc[
                        row["start"] - dt.timedelta(days=14) : row["start"]
                    ]
                )
            )
            .abs()
            .mean()
        )

        # Do the same for the two week period after the end of the event.
        stores_periods.loc[i, "2w_after_affected_fc_p_lim"] = (
            100
            * (
                (
                    n.links_t.p1[fc_i]
                    .loc[row["end"] : row["end"] + dt.timedelta(days=14)]
                    .abs()
                    > 0.99 * p_lims
                ).sum()
                > 0
            ).sum()
            / len(fc_i)
        )
        stores_periods.loc[i, "2w_after_ratio_p_lim"] = (
            100
            * (
                (
                    (
                        n.links_t.p1[fc_i]
                        .loc[row["end"] : row["end"] + dt.timedelta(days=14)]
                        .abs()
                        > 0.99 * p_lims
                    ).sum()
                )
                / len(
                    n.links_t.p1[fc_i].loc[
                        row["end"] : row["end"] + dt.timedelta(days=14)
                    ]
                )
            )
            .abs()
            .mean()
        )

        # Recovery costs of batteries.
        batt_i = n.links.loc[n.links.carrier == "battery charger"].index
        batt_costs = (
            n.links.loc[batt_i, "capital_cost"] * n.links.loc[batt_i, "p_nom_opt"]
        )
        batt_costs_e = (n.stores.capital_cost * n.stores.e_nom_opt).filter(
            like="battery"
        )
        batt_costs_e.index = batt_i
        batt_p = -n.links_t.p1.loc[:, batt_i] * all_prices.loc[n.snapshots].values
        stores_periods.loc[i, "recovered_battery_costs"] = (
            batt_p.loc[row["start"] : row["end"]].sum().sum()
            / (batt_costs + batt_costs_e).sum()
        )

        # Recovery costs of H2 storage.
        fc_i = n.links.loc[n.links.carrier == "H2 fuel cell"].index
        elec_i = n.links.loc[n.links.carrier == "H2 electrolysis"].index
        h2_i = n.stores.loc[n.stores.carrier == "H2"].index
        h2_costs = n.stores.loc[h2_i, "capital_cost"] * n.stores.loc[h2_i, "e_nom_opt"]
        fc_costs = n.links.loc[fc_i, "capital_cost"] * n.links.loc[fc_i, "p_nom_opt"]
        elec_costs = (
            n.links.loc[elec_i, "capital_cost"] * n.links.loc[elec_i, "p_nom_opt"]
        )
        elec_costs.index = fc_i
        h2_costs.index = fc_i
        fc_p = -n.links_t.p1.loc[:, fc_i] * all_prices.loc[n.snapshots].values
        stores_periods.loc[i, "recovered_h2_costs"] = (
            fc_p.loc[row["start"] : row["end"]].sum().sum()
            / (fc_costs + elec_costs + h2_costs).sum()
        )

    stores_periods["discharge"] = stores_periods["discharge"] / 1e6
    stores_periods = stores_periods.round(2)
    stores_periods["empty"] = stores_periods["empty"].round(1)

    return stores_periods


def anomaly_index_shift(df_origin, avg_df, years):
    """
    Adjusts the values in the original DataFrame by subtracting the corresponding values
    from the average DataFrame, with a shift in the year index based on the specified years.

    Parameters:
    df_origin (pd.DataFrame): The original DataFrame containing the data to be adjusted.
    avg_df (pd.DataFrame): The DataFrame containing the average values to be subtracted.
    years (list of int): A list containing the years to be used for shifting the index.
                         The first element is used as the base year for the shift.

    Returns:
    pd.DataFrame: A new DataFrame with the adjusted values.
    """
    df_end = df_origin.copy()
    shift_year = years[0]
    for i in df_end.index:
        shifted_index = str(i)
        if i.month < 7:
            shifted_index = str(shift_year + 1) + shifted_index[4:]
        else:
            shifted_index = str(years[0]) + shifted_index[4:]
        df_end.loc[i] = df_origin.loc[i] - avg_df.loc[shifted_index]
    return df_end

def fc_duration_curve(
    opt_networks: dict,
) -> pd.DataFrame:
    """
    Gather duration curve of fuel cell discharge from networks.

    Parameters:
    -----------
    opt_networks: dict
        Dictionary with optimal networks.

    Returns:
    --------
    pd.DataFrame
        DataFrame with fuel cell discharge for each year.
    """
    years = list(opt_networks.keys())
    m = opt_networks[years[0]]
    fc_i = m.links[m.links.carrier == "H2 fuel cell"].index

    gen_fc_df = pd.DataFrame(index=range(8760))
    for year, n in opt_networks.items():
        gen_fc_df[year] = -n.links_t.p1[fc_i].sum(axis=1).sort_values(ascending=True).values
    gen_fc_df = (gen_fc_df/1e3).round(1)
    return gen_fc_df

def gather_cfs(
    opt_networks: Dict[int, pypsa.Network],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gather capacity factors for wind and solar generators from optimized networks.

    Parameters:
    -----------
    opt_networks: Dict[int, pypsa.Network]
        Dictionary of optimized PyPSA networks

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Wind capacity factors and solar capacity factors
    """
    ns_wind_cf = []
    ns_solar_cf = []
    for year, n in opt_networks.items():
        wind_i = n.generators.index[
            n.generators.carrier.isin(
                ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]
            )
        ]
        solar_i = n.generators.index[n.generators.carrier.isin(["solar", "solar-hsat"])]

        ns_wind_cf.append(n.generators_t.p_max_pu.loc[:, wind_i])
        ns_solar_cf.append(n.generators_t.p_max_pu.loc[:, solar_i])
    wind_cf = pd.concat(ns_wind_cf, axis=0)
    solar_cf = pd.concat(ns_solar_cf, axis=0)
    return wind_cf, solar_cf

def gather_flex(
    opt_networks: dict,
) -> pd.DataFrame:
    """Gather duration curve of flexible dispatch from networks.
    
    Parameters:
    -----------
    opt_networks: dict
        Dictionary with optimal networks.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with flexible dispatch for each year."""
    # Get indices of dispatchable technologies.
    years = list(opt_networks.keys())
    m = opt_networks[years[0]]
    firm_gen_i = m.generators[m.generators.carrier.isin(["nuclear", "biomass", "ror", "OCGT", "CCGT"])].index
    firm_su_i = m.storage_units[m.storage_units.carrier.isin(["PHS", "hydro"])].index
    firm_s_i = m.links[m.links.carrier.isin(["battery discharger", "H2 fuel cell"])].index

    firm_flex_df = pd.DataFrame(index = range(8760))
    for year, n in opt_networks.items():
        df_helper = pd.DataFrame(index = range(8760), columns=["prod"])
        df_helper["prod"] = n.generators_t.p[firm_gen_i].sum(axis=1).values
        df_helper["prod"] += n.storage_units_t.p[firm_su_i].sum(axis=1).values
        df_helper["prod"] -= n.links_t.p1[firm_s_i].sum(axis=1).values
        firm_flex_df[year] = df_helper.sort_values(by="prod", ascending=False).values
    firm_flex_df = (firm_flex_df/1e3).round(1)
    return firm_flex_df

def gen_shares(
        n: pypsa.Network, 
        start: str = None, 
        end: str = None
        ) -> pd.DataFrame:
    """Calculate generation shares for all technologies in a network.
    
    Parameters:
    -----------
    n: pypsa.Network
        PyPSA network.
    start: str
        Start of period to be considered.
    end: str
        End of period to be considered.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with generation shares for each technology."""

    gen = n.generators_t.p
    gen.columns = n.generators.index.map(n.generators.carrier)
    gen = gen.groupby(axis=1, level=0).sum()

    su = n.storage_units_t.p.clip(lower=0)
    su.columns = n.storage_units.index.map(n.storage_units.carrier)
    su = su.groupby(axis=1, level=0).sum()

    links_i = n.links.loc[n.links.carrier.isin(["battery discharger", "H2 fuel cell"])].index
    link = -n.links_t.p1[links_i]
    link.columns = links_i.map(n.links.carrier)
    link = link.groupby(axis=1, level=0).sum()

    all_gen = pd.concat([gen, su, link], axis=1)

    all_gen = all_gen.loc[start:end]
    all_gen = all_gen.div(all_gen.sum(axis=1), axis=0)

    all_gen["solar"] = all_gen["solar"] + all_gen["solar-hsat"]
    all_gen = all_gen.drop(columns=["solar-hsat"])
    all_gen["offwind"] = all_gen["offwind-ac"] + all_gen["offwind-dc"] + all_gen["offwind-float"]
    all_gen = all_gen.drop(columns=["offwind-ac", "offwind-dc", "offwind-float"])
    return all_gen

def gen_stack_around_sde(
    opt_networks: Dict[int, pypsa.Network],
    periods: pd.DataFrame,
    window: pd.Timedelta = pd.Timedelta("14D"),
) -> pd.DataFrame:
    """
    Generate generation stacks for periods around system-defining events.

    Parameters:
    -----------
    opt_networks: Dict[int, pypsa.Network]
        Dictionary of optimized PyPSA networks
    periods: pd.DataFrame
        DataFrame with start and end times of system-defining events
    window: pd.Timedelta, default "14D"
        Time window to consider before and after each event

    Returns:
    --------
    pd.DataFrame
        Generation stacks for all technologies around system-defining events
    """
    gen_stacks = []
    for i, period in periods.iterrows():
        gen_stack = pd.DataFrame(
            columns=[
                "biomass",
                "nuclear",
                "ror",
                "battery discharger",
                "hydro",
                "PHS",
                "H2 fuel cell",
                "solar",
                "offwind",
                "onwind",
            ]
        ).astype(float)
        start_window = period.start - window
        end_window = period.end + window
        gen_stack = pd.DataFrame(
            index=pd.date_range(start_window, end_window, freq="h"),
            columns=[
                "biomass",
                "nuclear",
                "ror",
                "battery discharger",
                "battery charger",
                "hydro",
                "PHS",
                "H2 fuel cell",
                "H2 electrolysis",
                "solar",
                "offwind",
                "onwind",
            ],
        ).astype(float)
        n = opt_networks[get_net_year(period.start)]
        for tech in gen_stack.columns:
            if tech == "offwind":
                c_id = n.generators.loc[
                    n.generators.carrier.str.contains("offwind")
                ].index
                gen_stack.loc[start_window:end_window, tech] = n.generators_t.p.loc[
                    start_window:end_window, c_id
                ].sum(axis="columns")
            elif tech == "solar":
                c_id = n.generators.loc[
                    n.generators.carrier.str.contains("solar")
                ].index
                gen_stack.loc[start_window:end_window, tech] = n.generators_t.p.loc[
                    start_window:end_window, c_id
                ].sum(axis="columns")
            elif tech in ["H2 fuel cell", "battery discharger"]:
                l_id = n.links.loc[n.links.carrier == tech].index
                gen_stack.loc[start_window:end_window, tech] = -n.links_t.p1.loc[
                    start_window:end_window, l_id
                ].sum(axis="columns")
            elif tech in ["H2 electrolysis", "battery charger"]:
                l_id = n.links.loc[n.links.carrier == tech].index
                gen_stack.loc[start_window:end_window, tech] = -n.links_t.p0.loc[
                    start_window:end_window, l_id
                ].sum(axis="columns")
            elif tech in ["PHS", "hydro"]:
                su_id = n.storage_units.loc[n.storage_units.carrier == tech].index
                gen_stack.loc[start_window:end_window, tech] = n.storage_units_t.p.loc[
                    start_window:end_window, su_id
                ].sum(axis="columns")
            else:
                c_id = n.generators.loc[n.generators.carrier == tech].index
                gen_stack.loc[start_window:end_window, tech] = n.generators_t.p.loc[
                    start_window:end_window, c_id
                ].sum(axis="columns")
        gen_stacks.append(gen_stack)
    gen_stacks = pd.concat(gen_stacks)
    return gen_stacks


def nodal_flex_periods_seasonality(
    nodal_flex_u_xr,
    nodes,
    years,
    periods,
):
    """
    Calculate the seasonality, period usage, anomaly, and peak anomaly of nodal flexibility usage.

    Parameters:
    nodal_flex_u_xr (xarray.DataArray): The nodal flexibility usage data.
    nodes (list): List of nodes to analyze.
    years (list): List of years to consider for averaging.
    periods (pandas.DataFrame): DataFrame containing the start and end times of significant demand events (SDEs).

    Returns:
    tuple: A tuple containing the following DataFrames:
        - nodal_seasonality (pandas.DataFrame): Seasonality of flexibility usage per node.
        - nodal_period_flex_u (pandas.DataFrame): Flexibility usage per node during SDEs.
        - nodal_anomaly_flex_u (pandas.DataFrame): Anomaly of flexibility usage per node during SDEs.
        - nodal_peak_anomaly_flex_u (pandas.DataFrame): Anomaly of flexibility usage per node during peak hours of SDEs.
    """
    (
        nodal_seasonality,
        nodal_period_flex_u,
        nodal_anomaly_flex_u,
        nodal_peak_anomaly_flex_u,
    ) = {}, {}, {}, {}
    for node in nodes:
        # Seasonality of usage of flexibility per node
        nfu = (
            nodal_flex_u_xr.sel(node=node)
            .to_pandas()
            .drop("node", axis="columns")
            .astype(float)
            .round(2)
        )
        nodal_seasonality[node] = average_across_years(nfu, years)
        # Usage of flexibility per node during SDEs
        nodal_period_flex_u[node] = pd.concat(
            [nfu.loc[row["start"] : row["end"]] for i, row in periods.iterrows()],
        )
        # Anomaly of usage of flexibility per node during SDEs
        nodal_anomaly_u = nodal_period_flex_u[node].copy()
        nodal_anomaly_flex_u[node] = anomaly_index_shift(
            nodal_anomaly_u, nodal_seasonality[node], years
        )
        # Anomaly of usage of flexibility per node during peak hours of SDEs
        nodal_peak_anomaly_flex_u[node] = anomaly_index_shift(
            pd.concat(
                [nfu.loc[row["peak_hour"]] for i, row in periods.iterrows()], axis=1
            ).T,
            nodal_seasonality[node],
            years,
        )
    nodal_seasonality = (
        pd.concat(nodal_seasonality, axis="columns").astype(float).round(2)
    )
    nodal_period_flex_u = (
        pd.concat(nodal_period_flex_u, axis="index").astype(float).round(2)
    )
    nodal_anomaly_flex_u = (
        pd.concat(nodal_anomaly_flex_u, axis="index").astype(float).round(2)
    )
    nodal_peak_anomaly_flex_u = (
        pd.concat(nodal_peak_anomaly_flex_u, axis="index").astype(float).round(2)
    )
    return (
        nodal_seasonality,
        nodal_period_flex_u,
        nodal_anomaly_flex_u,
        nodal_peak_anomaly_flex_u,
    )

def peak_hour_gen(
    opt_networks: dict,
    periods: pd.DataFrame,
    net_load: pd.DataFrame,
):
    """
    Returns the generation during the peak hour of the SDEs.
    
    Parameters:
    -----------
    opt_networks: dict
        Dictionary of PyPSA networks.
    periods: pd.DataFrame
        System-defining events.
    net_load: pd.DataFrame
        Net load dataframe."""
    carrier_tech = ["biomass", "nuclear", "offwind", "solar", "onwind", "ror"]
    links_tech = ["H2 fuel cell", "battery discharger"]
    su_tech = ["PHS", "hydro"]

    peak_gen = pd.DataFrame(columns = carrier_tech + links_tech + su_tech + ["net load"], index = periods.peak_hour).astype(float)

    for period in periods.index:
        peak_hour = periods.loc[period, "peak_hour"]
        net_year = get_net_year(peak_hour)
        n = opt_networks[net_year]
        
        for tech in carrier_tech:
            if tech == "offwind":
                c_id = n.generators.loc[n.generators.carrier.str.contains("offwind")].index
            elif tech == "solar":
                c_id = n.generators.loc[n.generators.carrier.str.contains("solar")].index
            else:
                c_id = n.generators.loc[n.generators.carrier == tech].index
            peak_gen.loc[peak_hour,tech] = n.generators_t.p.loc[peak_hour, c_id].sum()
        for tech in links_tech:
            l_id = n.links.loc[n.links.carrier == tech].index
            peak_gen.loc[peak_hour,tech] = n.links_t.p1.loc[peak_hour, l_id].abs().sum()
        for tech in su_tech:
            su_id = n.storage_units.loc[n.storage_units.carrier == tech].index
            peak_gen.loc[peak_hour,tech] = n.storage_units_t.p.loc[peak_hour, su_id].sum()
        peak_gen.loc[peak_hour,"net load"] = net_load.loc[peak_hour]

    peak_gen /= 1e3 # in GW
    peak_gen = peak_gen.round(1)
    return peak_gen

def solar_capacity_factors(opt_networks, years, winter=False):
    """
    Calculate solar capacity factors for given optimized networks over specified years.
    Parameters:
    opt_networks (dict): A dictionary where keys are years and values are network objects containing generator data.
    years (list): A list of years for which to calculate the capacity factors.
    winter (bool, optional): If True, calculate capacity factors only for winter months (October to March). Defaults to False.
    Returns:
    pd.DataFrame: A DataFrame with years as index and column ["solar"] containing the capacity factors.
    """
    df = pd.DataFrame(index=years, columns=["solar"])

    for year, n in opt_networks.items():
        # Solar (including solar-hsat)
        solar_i = n.generators.loc[
            n.generators.carrier.isin(["solar", "solar-hsat"])
        ].index

        if winter:
            df.loc[year, "solar"] = n.generators_t.p[solar_i].loc[
                n.generators_t.p[solar_i].index.month.isin([10, 11, 12, 1, 2, 3])
            ].sum().sum() / (n.generators.loc[solar_i, "p_nom_opt"].sum() * 4380)
        else:
            df.loc[year, "solar"] = n.generators_t.p[solar_i].sum().sum() / (
                n.generators.loc[solar_i, "p_nom_opt"].sum() * 8760
            )
    return df


def stats_sde(
    periods: pd.DataFrame,
    stores_periods: pd.DataFrame,
    net_load: pd.Series,
    total_load: pd.Series,
    avg_load: pd.DataFrame,
    avg_wind: pd.DataFrame,
    wind_cf: pd.DataFrame,
    wind_caps: pd.DataFrame,
    reindex_opt_objs: pd.Series,
    total_costs: Dict[int, pd.DataFrame],
    all_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistical metrics for system-defining events.

    Parameters:
    -----------
    periods: pd.DataFrame
        DataFrame with start and end times of system-defining events
    stores_periods: pd.DataFrame
        DataFrame with storage-related characteristics of SDEs
    net_load: pd.Series
        Time series of net load
    total_load: pd.Series
        Time series of total load
    avg_load: pd.DataFrame
        Average load per time period
    avg_wind: pd.DataFrame
        Average wind generation per time period
    wind_cf: pd.DataFrame
        Wind capacity factors
    wind_caps: pd.DataFrame
        Wind capacities
    reindex_opt_objs: pd.Series
        Reindexed objective function values
    total_costs: Dict[int, pd.DataFrame]
        Dictionary of total costs per year
    all_prices: pd.DataFrame
        DataFrame with electricity prices

    Returns:
    --------
    pd.DataFrame
        DataFrame with statistical metrics for system-defining events
    """
    stats_periods = periods.copy()
    for i in stats_periods.index:
        stats_periods.loc[i, "net_load_peak_hour"] = (
            net_load.loc[stats_periods.loc[i, "peak_hour"]] / 1e3
        )
        stats_periods.loc[i, "highest_net_load"] = (
            net_load.loc[
                stats_periods.loc[i, "start"] : stats_periods.loc[i, "end"]
            ].max()
            / 1e3
        )
        stats_periods.loc[i, "avg_net_load"] = (
            net_load.loc[
                stats_periods.loc[i, "start"] : stats_periods.loc[i, "end"]
            ].mean()
            / 1e3
        )
        stats_periods.loc[i, "energy_deficit"] = (
            net_load.loc[
                stats_periods.loc[i, "start"] : stats_periods.loc[i, "end"]
            ].sum()
            / 1e6
        )
        stats_periods.loc[i, "duration"] = (
            pd.Timestamp(stats_periods.loc[i, "end"])
            - pd.Timestamp(stats_periods.loc[i, "start"])
        ).total_seconds() / 3600
        stats_periods.loc[i, "h2_discharge"] = stores_periods.loc[
            stores_periods.index[i], "discharge"
        ]
        stats_periods.loc[i, "max_fc_discharge"] = stores_periods.loc[
            stores_periods.index[i], "max_fc_discharge"
        ]
        start, end = stats_periods.loc[i, "start"], stats_periods.loc[i, "end"]
        year_s = 1942 if start.month < 7 else 1941
        year_e = 1942 if end.month < 7 else 1941
        shifted_start = pd.Timestamp(
            f"{year_s}-{str(start.month).zfill(2)}-{str(start.day).zfill(2)} {str(start.hour).zfill(2)}:00:00"
        )
        shifted_end = pd.Timestamp(
            f"{year_e}-{str(end.month).zfill(2)}-{str(end.day).zfill(2)} {str(end.hour).zfill(2)}:00:00"
        )
        helper_df = total_load.loc[start:end].values / avg_load.loc[
            shifted_start:shifted_end
        ].sum(axis=1)
        helper_df = pd.DataFrame(helper_df, index=pd.date_range(start, end, freq="h"))
        stats_periods.loc[i, "avg_rel_load"] = np.mean(
            (total_load.loc[start:end])
            / (avg_load.loc[shifted_start:shifted_end].sum(axis=1).values)
        )
        n_year = get_net_year(start)
        stats_periods.loc[i, "wind_cf"] = (
            ((wind_cf @ wind_caps[n_year]) / wind_caps[n_year].sum())
            .loc[start:end]
            .mean()
        )
        wind_anom = (
            stats_periods.loc[i, "wind_cf"]
            - ((avg_wind @ wind_caps[n_year]) / wind_caps[n_year].sum())
            .loc[shifted_start:shifted_end]
            .mean()
        )
        stats_periods.loc[i, "wind_anom"] = wind_anom
        stats_periods = stats_periods.round(3)
        stats_periods[
            [
                "net_load_peak_hour",
                "highest_net_load",
                "avg_net_load",
            ]
        ] = stats_periods[
            ["net_load_peak_hour", "highest_net_load", "avg_net_load"]
        ].round(0)
        net_y = int(get_net_year(periods.loc[i, "start"]))
        stats_periods.loc[i, "annual_cost"] = int(reindex_opt_objs.loc[net_y] / 1e9)
        stats_periods.loc[i, "recovered_battery_costs"] = stores_periods.loc[
            stores_periods.index[i], "recovered_battery_costs"
        ]
        stats_periods.loc[i, "recovered_h2_costs"] = stores_periods.loc[
            stores_periods.index[i], "recovered_h2_costs"
        ]
        stats_periods.loc[i, "total_costs"] = (
            total_costs[net_y].loc[start:end].sum() / 1e9
        )
        stats_periods.loc[i, "share_total_costs"] = stats_periods.loc[
            i, "total_costs"
        ] / (total_costs[net_y].sum().sum() / 1e9)
        stats_periods.loc[i, "avg_hourly_cost"] = (
            total_costs[net_y].loc[start:end].mean() / 1e6
        )
        stats_periods.loc[i, "normed_price_std"] = (
            (all_prices.loc[start:end].mean()) / all_prices.loc[start:end].mean().max()
        ).std()

    return stats_periods


def wind_capacity_factors(opt_networks, years, winter=False):
    """
    Calculate wind capacity factors for given optimized networks over specified years.
    Parameters:
    opt_networks (dict): A dictionary where keys are years and values are network objects containing generator data.
    years (list): A list of years for which to calculate the capacity factors.
    winter (bool, optional): If True, calculate capacity factors only for winter months (October to March). Defaults to False.
    Returns:
    pd.DataFrame: A DataFrame with years as index and columns ["wind", "onwind", "offwind"] containing the capacity factors.
    """
    df = pd.DataFrame(index=years, columns=["wind", "onwind", "offwind"])

    for year, n in opt_networks.items():
        # Wind
        wind_i = n.generators.loc[n.generators.carrier.str.contains("wind")].index
        # Onshore wind
        onwind_i = n.generators.loc[n.generators.carrier == "onwind"].index
        # Offshore wind
        offwind_i = n.generators.loc[n.generators.carrier.str.contains("offwind")].index
        if winter:
            df.loc[year, "wind"] = n.generators_t.p[wind_i].loc[
                n.generators_t.p[wind_i].index.month.isin([10, 11, 12, 1, 2, 3])
            ].sum().sum() / (n.generators.loc[wind_i, "p_nom_opt"].sum() * 4380)
            df.loc[year, "onwind"] = n.generators_t.p[onwind_i].loc[
                n.generators_t.p[onwind_i].index.month.isin([10, 11, 12, 1, 2, 3])
            ].sum().sum() / (n.generators.loc[onwind_i, "p_nom_opt"].sum() * 4380)
            df.loc[year, "offwind"] = n.generators_t.p[offwind_i].loc[
                n.generators_t.p[offwind_i].index.month.isin([10, 11, 12, 1, 2, 3])
            ].sum().sum() / (n.generators.loc[offwind_i, "p_nom_opt"].sum() * 4380)
        else:
            df.loc[year, "wind"] = n.generators_t.p[wind_i].sum().sum() / (
                n.generators.loc[wind_i, "p_nom_opt"].sum() * 8760
            )

            df.loc[year, "onwind"] = n.generators_t.p[onwind_i].sum().sum() / (
                n.generators.loc[onwind_i, "p_nom_opt"].sum() * 8760
            )

            df.loc[year, "offwind"] = n.generators_t.p[offwind_i].sum().sum() / (
                n.generators.loc[offwind_i, "p_nom_opt"].sum() * 8760
            )
    return df


def wind_dist(
    opt_networks,
):
    """
    Calculate the daily mean wind power distribution for specified months over multiple years.

    Parameters:
    opt_networks (dict): A dictionary where keys are years and values are network objects. Each network object should have:
        - generators (DataFrame): A DataFrame containing generator information, including a 'carrier' column to identify wind generators and a 'p_nom_opt' column for nominal power.
        - generators_t (DataFrame): A DataFrame containing time series data of generator power output ('p').

    Returns:
    pd.DataFrame: A concatenated DataFrame of daily mean wind power distribution for the specified months (October to March) across all years in the input dictionary.
    """

    wind_distr = {}
    for year, n in opt_networks.items():
        wind_i = n.generators.loc[n.generators.carrier.str.contains("wind")].index
        wind_distr[year] = (
            n.generators_t.p[wind_i]
            .loc[n.generators_t.p[wind_i].index.month.isin([10, 11, 12, 1, 2, 3])]
            .sum(axis="columns")
            .resample("1D")
            .mean()
            / n.generators.loc[wind_i, "p_nom_opt"].sum()
        )
    return pd.concat(wind_distr)


if __name__ == "__main__":
    # Standard set-up, note the faulty CO2 limit naming (as no CO2-emitting technologies were included)
    config_name = "stressful-weather"
    config_str = "base_s_90_elec_lc1.25_Co2L"

    folder = f"./processing_data/{config_name}"

    # Load optimal networks
    config, scenario_def, years, opt_networks = load_opt_networks(
        config_name, config_str=config_str, load_networks=True
    )

    # Use a standard network for easing notation.
    m = opt_networks[years[0]]

    # Load system-defining events
    periods = load_periods(config)

    ## OBJECTIVE VALUES
    opt_objs, _ = optimal_costs(opt_networks)
    opt_objs.round(0).to_csv(f"{folder}/opt_objs.csv")
    reindex_opt_objs = opt_objs.copy().sum(axis="columns")
    reindex_opt_objs.index = years

    ## MEANS
    avg_load = pd.read_csv(
        "../pypsa-eur/results/means/load_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv",
        index_col=0,
        parse_dates=True,
    )
    avg_wind = pd.read_csv(
        "../pypsa-eur/results/means/wind_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv",
        index_col=0,
        parse_dates=True,
    )
    avg_solar = pd.read_csv(
        "../pypsa-eur/results/means/solar_1941-2020_100bn_12-336h_90_elec_lc1.25_Co2L.csv",
        index_col=0,
        parse_dates=True,
    )

    ## LOAD

    # Net load
    net_load = compute_net_load(opt_networks, m)
    net_load.round(0).to_csv(f"{folder}/net_load.csv")

    # Total load
    total_load = pd.concat(
        [n.loads_t.p.sum(axis="columns") for n in opt_networks.values()]
    )
    total_load.round(0).to_csv(f"{folder}/total_load.csv")
    nodal_load = pd.concat(
        [n.loads_t.p.round(0) for n in opt_networks.values()], axis=0
    )
    nodal_load.to_csv(f"{folder}/nodal_load.csv")

    # Winter load
    winter_load = pd.DataFrame(index=years, columns=["load"])
    for year, n in opt_networks.items():
        winter_load.loc[year, "load"] = (
            n.loads_t.p.sum(axis="columns")
            .loc[n.loads_t.p.index.month.isin([10, 11, 12, 1, 2, 3])]
            .mean()
            / 1e3
        )
    winter_load.round(1).to_csv(f"{folder}/winter_load.csv")

    ## COSTS

    # All prices
    price_nodes = m.buses[m.buses.carrier == "AC"].index
    all_prices = pd.concat(
        [n.buses_t.marginal_price.loc[:, price_nodes] for n in opt_networks.values()],
        axis=0,
    )
    all_prices.round(0).to_csv(f"{folder}/all_prices.csv")

    # ACCUMULATION OF COSTS DURING PERIOD AND WINTER
    periods_cost = cost_acc(opt_networks, periods=periods)
    winter_costs = cost_acc(opt_networks, years=years)
    periods_cost.to_csv(f"{folder}/periods_cost.csv")
    winter_costs.to_csv(f"{folder}/winter_costs.csv")

    # Costs, storage costs and fuel cell costs
    total_costs, total_storage_costs, total_fc_costs = compute_all_duals(opt_networks)
    pd.concat(total_costs).to_csv(f"{folder}/total_costs.csv")
    pd.concat(total_storage_costs).to_csv(f"{folder}/total_storage_costs.csv")
    pd.concat(total_fc_costs).to_csv(f"{folder}/total_fc_costs.csv")

    ## STORAGE
    # Storage capacities
    s_caps = pd.DataFrame(index=years, columns=["H2_e", "FC_p", "EL_p", "batt_e", "batt_c_p", "batt_d_p", "PHS_e", "hydro_e"])
    for year, n in opt_networks.items():
        # Hydrogen storage
        s_caps.loc[year, "H2_e"] = (
            n.stores.loc[
                n.stores.loc[n.stores.carrier == "H2"].index, "e_nom_opt"
            ].sum()
            / 1e3
        )  # GWh
        fc_i = n.links.loc[n.links.carrier == "H2 fuel cell"].index
        s_caps.loc[year, "FC_p"] = (
            (n.links.loc[
                fc_i, "p_nom_opt"
            ] * n.links.loc[fc_i, "efficiency"]).sum()
            / 1e3
        )  # GW
        elec_i = n.links.loc[n.links.carrier == "H2 electrolysis"].index
        s_caps.loc[year, "EL_p"] = (
            (n.links.loc[
                elec_i, "p_nom_opt"
            ] * n.links.loc[elec_i, "efficiency"]).sum()
            / 1e3
        )  # GW
        # Battery storage
        s_caps.loc[year, "batt_e"] = (
            n.stores.loc[
                n.stores.loc[n.stores.carrier == "battery"].index, "e_nom_opt"
            ].sum()
            / 1e3
        )  # GWh
        batt_c_i = n.links.loc[n.links.carrier == "battery charger"].index
        s_caps.loc[year, "batt_c_p"] = (
            (n.links.loc[
                batt_c_i, "p_nom_opt"
            ] * n.links.loc[batt_c_i, "efficiency"]).sum()
            / 1e3
        )  # GW
        batt_d_i = n.links.loc[n.links.carrier == "battery discharger"].index
        s_caps.loc[year, "batt_d_p"] = (
            (n.links.loc[
                batt_d_i, "p_nom_opt"
            ] * n.links.loc[batt_d_i, "efficiency"]).sum()
            / 1e3
        )
        # Pumped hydro storage
        phs_i = n.storage_units.loc[n.storage_units.carrier == "PHS"].index
        s_caps.loc[year, "PHS_e"] = (n.storage_units.loc[phs_i, "p_nom"] * n.storage_units.loc[phs_i, "max_hours"]).sum()/1e3 # GWh
        # Hydro storage
        hydro_i = n.storage_units.loc[n.storage_units.carrier == "hydro"].index
        s_caps.loc[year, "hydro_e"] = (n.storage_units.loc[hydro_i, "p_nom"] * n.storage_units.loc[hydro_i, "max_hours"]).sum()/1e3 # GWh
    s_caps.round(2).to_csv(f"{folder}/s_caps.csv")

    # Average storage levels
    su_i = m.storage_units.index
    su_soc = pd.concat(
        [n.storage_units_t.state_of_charge[su_i] for n in opt_networks.values()],
        axis="rows",
    ).round(0)
    su_soc.to_csv(f"{folder}/state_of_charge.csv")
    avg_soc = average_across_years(su_soc, years).round(0)
    avg_soc.to_csv(f"{folder}/avg_soc.csv")

    ## TRANSMISSION

    ## CAPACITY FACTORS
    wind_cf, solar_cf = gather_cfs(opt_networks)
    wind_cf.round(4).to_xarray().to_netcdf(f"{folder}/wind_cf.nc")
    solar_cf.round(4).to_xarray().to_netcdf(f"{folder}/solar_cf.nc")

    # Capacities of technologies
    wind_i = m.generators.loc[
        m.generators.carrier.isin(
            ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]
        )
    ].index
    wind_caps = pd.concat(
        [n.generators.loc[wind_i, "p_nom_opt"] for n in opt_networks.values()],
        axis="columns",
        keys=years,
    ).round(0)
    wind_caps.to_csv(f"{folder}/wind_caps.csv")
    solar_i = m.generators.loc[m.generators.carrier.isin(["solar", "solar-hsat"])].index
    solar_caps = pd.concat(
        [n.generators.loc[solar_i, "p_nom_opt"] for n in opt_networks.values()],
        axis="columns",
        keys=years,
    ).round(0)
    solar_caps.to_csv(f"{folder}/solar_caps.csv")

    # Wind distribution across years
    wind_distr = wind_dist(opt_networks)
    wind_distr.round(4).to_csv(f"{folder}/wind_distr.csv")

    # Hydro, costs, wind capacity factors across years
    annual_inflow = pd.DataFrame()
    for year, n in opt_networks.items():
        annual_inflow.loc[year, "Annual inflow"] = n.storage_units_t.inflow.sum().sum()
    annual_inflow.round(0).to_csv(f"{folder}/annual_inflow.csv")

    annual_cfs = wind_capacity_factors(opt_networks, years, winter=False)
    winter_cfs = wind_capacity_factors(opt_networks, years, winter=True)
    annual_solar_cfs = solar_capacity_factors(opt_networks, years, winter=False)
    winter_solar_cfs = solar_capacity_factors(opt_networks, years, winter=True)
    annual_cfs = pd.concat([annual_cfs, annual_solar_cfs], axis="columns")
    winter_cfs = pd.concat([winter_cfs, winter_solar_cfs], axis="columns")
    annual_cfs.round(4).to_csv(f"{folder}/annual_cfs.csv")
    winter_cfs.round(4).to_csv(f"{folder}/winter_cfs.csv")

    # Deficits, net load
    longest_deficit = compute_longest_deficit(years, net_load)
    longest_deficit.to_csv(f"{folder}/longest_net_load_deficit.csv")

    highest_deficit = pd.DataFrame(index=years, columns=["time", "value"])
    for year in years:
        highest_deficit.loc[year, "time"] = net_load.loc[
            f"{year}-07-01" : f"{year + 1}-06-30"
        ].idxmax()
        highest_deficit.loc[year, "value"] = float(
            int(net_load.loc[f"{year}-07-01" : f"{year + 1}-06-30"].max())
        )
    highest_deficit.to_csv(f"{folder}/highest_net_load_deficit.csv")

    weekly_net_load = net_load.rolling(168).sum()
    max_weekly_net_load = pd.DataFrame(
        index=years, columns=["start", "end", "total", "mean"]
    )
    for y in years:
        max_weekly_net_load.loc[y, "end"] = weekly_net_load.loc[
            f"{y}-07-01" : f"{y + 1}-06-30"
        ].idxmax()
        max_weekly_net_load.loc[y, "start"] = max_weekly_net_load.loc[
            y, "end"
        ] - dt.timedelta(days=7)
        max_weekly_net_load.loc[y, "total"] = weekly_net_load.loc[
            f"{y}-07-01" : f"{y + 1}-06-30"
        ].max()
        max_weekly_net_load.loc[y, "mean"] = (
            weekly_net_load.loc[max_weekly_net_load.loc[y, "end"]] / 168
        )
    max_weekly_net_load.round(0).to_csv(f"{folder}/max_weekly_net_load.csv")

    ## SYSTEM-DEFINING EVENTS
    # Stats:
    stores_periods = characteristics_sdes(opt_networks, periods, all_prices)
    stores_periods.to_csv(f"{folder}/stores_periods.csv")

    # Stats for clustering: net load peak hour, highest net load, avg net load, energy deficit, h2 discharge, max fc discharge, avg rel load, wind cf, wind anom, annual cost, recovered battery costs, recovered h2 costs, total costs, share total costs, avg hourly cost, normed price std
    stats_periods = stats_sde(
        periods,
        stores_periods,
        net_load,
        total_load,
        avg_load,
        avg_wind,
        wind_cf,
        wind_caps,
        reindex_opt_objs,
        total_costs,
        all_prices,
    )
    stats_periods.to_csv(f"{folder}/stats_periods.csv")

    # Generation with two week windows around SDEs.
    gen_stacks = gen_stack_around_sde(opt_networks, periods, window=pd.Timedelta("14D"))
    gen_stacks.round(0).to_csv(f"{folder}/gen_stacks.csv")

    # Peak hour generation during SDEs
    peak_hour_gen_df = peak_hour_gen(opt_networks, periods, net_load)
    peak_hour_gen_df.round(1).to_csv(f"{folder}/peak_gen.csv")


    ## FLEXIBILITY INDICATORS
    ### SYSTEM
    # Detailed system flexibility
    all_flex_detailed = pd.concat(
        [detailed_system_flex(n) for n in opt_networks.values()]
    )

    (
        avg_flex_detailed,
        periods_flex_detailed,
        periods_anomaly_flex_detailed,
        periods_peak_flex_detailed,
        periods_peak_anomaly_flex_detailed,
    ) = compute_flex_periods_anomalies(all_flex_detailed, periods, years)
    all_flex_detailed.round(3).to_csv(f"{folder}/all_flex_detailed.csv")
    avg_flex_detailed.round(3).to_csv(f"{folder}/avg_flex_detailed.csv")
    periods_flex_detailed.round(3).to_csv(f"{folder}/periods_flex_detailed.csv")
    periods_anomaly_flex_detailed.round(3).to_csv(
        f"{folder}/periods_anomaly_flex_detailed.csv"
    )
    periods_peak_flex_detailed.round(3).to_csv(
        f"{folder}/periods_peak_flex_detailed.csv"
    )
    periods_peak_anomaly_flex_detailed.round(3).to_csv(
        f"{folder}/periods_peak_anomaly_flex_detailed.csv"
    )
    # Coarse system flexibility
    all_flex_coarse = pd.concat([coarse_system_flex(n) for n in opt_networks.values()])
    (
        avg_flex_coarse,
        periods_flex_coarse,
        periods_anomaly_flex_coarse,
        periods_peak_flex_coarse,
        periods_peak_anomaly_flex_coarse,
    ) = compute_flex_periods_anomalies(all_flex_coarse, periods, years)
    all_flex_coarse.round(3).to_csv(f"{folder}/all_flex_coarse.csv")
    avg_flex_coarse.round(3).to_csv(f"{folder}/avg_flex_coarse.csv")
    periods_flex_coarse.round(3).to_csv(f"{folder}/periods_flex_coarse.csv")
    periods_anomaly_flex_coarse.round(3).to_csv(
        f"{folder}/periods_anomaly_flex_coarse.csv"
    )
    periods_peak_flex_coarse.round(3).to_csv(f"{folder}/periods_peak_flex_coarse.csv")
    periods_peak_anomaly_flex_coarse.round(3).to_csv(
        f"{folder}/periods_peak_anomaly_flex_coarse.csv"
    )

    # Coarse flexibility during system-defining events

    ### NODAL
    nodes = list(m.buses.location.unique())
    nodes.pop(0)
    nfp, nfu = nodal_flexibility(opt_networks, nodes)
    nodal_flex_p = pd.concat(nfp)
    nodal_flex_p.to_csv(f"{folder}/nodal_flex_p.csv")
    # for node in nodes:
    #     nfu[node].to_csv(f"{folder}/nodal_flex_u/nodal_flex_u_{node}.csv")
    nodal_flex_u = {node: nfu[node] for node in nodes}
    nodal_flex_u_xr = xr.concat(
        [nodal_flex_u[node].to_xarray() for node in nodes],
        dim=pd.Index(nodes, name="node"),
    )
    nodal_flex_u_xr.to_netcdf(f"{folder}/nodal_flex_u.nc")
    (
        nodal_seasonality,
        nodal_flex_periods,
        nodal_flex_anomaly_periods,
        nodal_peak_anomaly_flex,
    ) = nodal_flex_periods_seasonality(nodal_flex_u_xr, nodes, years, periods)
    nodal_seasonality.to_csv(f"{folder}/nodal_seasonality.csv")
    nodal_flex_periods.to_csv(f"{folder}/nodal_periods_flex_u.csv")
    nodal_flex_anomaly_periods.to_csv(f"{folder}/nodal_anomaly_flex_u.csv")
    nodal_peak_anomaly_flex.to_csv(f"{folder}/nodal_peak_anomaly_flex_u.csv")

    ## SYSTEM VARIABLES
    # All system anomaly: Net load anomaly, load anom, wind anom, solar anom
    # For the disp_technologies, compute the used flexibility and the anomaly at all times.
    disp_tech = [
        "biomass",
        "nuclear",
        "H2 fuel cell",
        "battery discharger",
        "PHS",
        "hydro",
    ]
    all_system_anomaly = compute_system_anomaly(
        periods,
        years,
        total_load,
        wind_cf,
        solar_cf,
        avg_load,
        avg_wind,
        avg_solar,
        opt_networks,
    )
    all_used_flexibility = compute_used_flex(
        periods, nodal_flex_p, nodal_flex_periods, disp_tech
    )
    all_flex_anomaly = compute_used_flex(
        periods, nodal_flex_p, nodal_flex_anomaly_periods, disp_tech
    )

    all_system_anomaly.to_csv(f"{folder}/all_system_anomaly.csv")
    all_used_flexibility.to_csv(f"{folder}/all_used_flexibility.csv")
    all_flex_anomaly.to_csv(f"{folder}/all_flex_anomaly.csv")

    # Get duration curve
    firm_flex_df = gather_flex(opt_networks)
    firm_flex_df.to_csv(f"{folder}/firm_flex_df.csv")

    # Get duration curve of fuel cells.
    gen_fc_df = fc_duration_curve(opt_networks)
    gen_fc_df.to_csv(f"{folder}/gen_fc_df.csv")

    ## HYDRO
    # Hydro and PHS costs
    hydro_costs, phs_costs = compute_hydro_phs_costs(opt_networks)
    hydro_costs.round(0).to_csv(f"{folder}/hydro_costs.csv")
    phs_costs.round(0).to_csv(f"{folder}/phs_costs.csv")

    # GENERATION SHARES
    winter_shares = pd.DataFrame(index=years, columns = ['biomass', 'nuclear', 'offwind', 'onwind', 'ror', 'solar', 'PHS', 'hydro', 'H2 fuel cell','battery discharger'])
    periods_shares = pd.DataFrame(index=periods.index, columns = ['biomass', 'nuclear', 'offwind', 'onwind', 'ror', 'solar', 'PHS', 'hydro', 'H2 fuel cell', 'battery discharger'])
    annual_gen_shares = pd.DataFrame(index=years, columns = ['biomass', 'nuclear', 'offwind', 'onwind', 'ror', 'solar', 'PHS', 'hydro', 'H2 fuel cell','battery discharger'])
    nov_feb_shares = pd.DataFrame(index=years, columns = ['biomass', 'nuclear', 'offwind', 'onwind', 'ror', 'solar', 'PHS', 'hydro', 'H2 fuel cell','battery discharger'])
    for year, n in opt_networks.items():
        winter_shares.loc[year] = gen_shares(n, start=f"{year}-10-01", end=f"{year+1}-03-31").mean(axis=0)
        annual_gen_shares.loc[year] = gen_shares(n).mean(axis=0)
        nov_feb_shares.loc[year] = gen_shares(n, start=f"{year}-11-01", end=f"{year+1}-02-28").mean(axis=0)
    for i, period in periods.iterrows():
        n = opt_networks[get_net_year(period.start)]
        periods_shares.loc[i] = gen_shares(n, start=period.start, end=period.end).mean(axis=0)
    # Export
    winter_shares.to_csv(f"{folder}/generation_winter_shares.csv")
    periods_shares.to_csv(f"{folder}/generation_periods_shares.csv")
    annual_gen_shares.to_csv(f"{folder}/generation_annual_shares.csv")
    nov_feb_shares.to_csv(f"{folder}/generation_nov_feb_shares.csv")

    

