# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Gather functions that we use in the notebooks for analysing the networks.
Returns:
--------
float
    The annual variable costs in the network.
"""

import datetime as dt
import yaml
import pypsa

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Patch

import geopandas as gpd
import cartopy.crs as ccrs
import seaborn as sns

import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.dates as mdates
from typing import Optional

mpl.rcParams["figure.dpi"] = 150

## COMPUTATIONS AND INITIATION

cm = 1 / 2.54


def annual_variable_cost(
    n: pypsa.Network,
) -> float:
    """Compute the annual variable costs in a PyPSA network `n`. Don't count load shedding.

    Parameters:
    -----------
    n:  PyPSA network.
    """
    weighting = n.snapshot_weightings.objective
    total = 0
    # Add variable costs for generators
    i = n.generators.loc[n.generators.carrier != "load-shedding"].index
    total += (
        n.generators_t.p[i].multiply(weighting, axis=0).sum(axis=0)
        * n.generators.marginal_cost
    ).sum()
    # Add variable costs for links (lines have none), in our model all 0 though?
    total += (
        n.links_t.p0[n.links.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.links.marginal_cost
    ).sum()
    # Add variable costs for stores
    total += (
        n.stores_t.p[n.stores.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.stores.marginal_cost
    ).sum()
    # Add variable costs for storage units
    total += (
        n.storage_units_t.p[n.storage_units.index]
        .abs()
        .multiply(weighting, axis=0)
        .sum(axis=0)
        * n.storage_units.marginal_cost
    ).sum()
    # Divide by the number of years the network is defined over. We disregard
    # leap years.
    total /= n.snapshot_weightings.objective.sum() / 8760
    return total


def compute_all_duals(
    opt_networks: dict,
    storage_units: bool = False,
    by_node: bool = False,
):
    """Compute the duals for the electricity costs, storage costs and fuel cell costs.

    Parameters:
    -----------
    opt_networks: dict
        Dictionary of PyPSA networks.
    storage_units: bool
        If True, storage units are considered, otherwise stores.
    by_node: bool
        If True, return the costs by node.
    """
    years = list(opt_networks.keys())
    # Electricity costs
    nodal_costs = {
        y: opt_networks[y].buses_t["marginal_price"] * opt_networks[y].loads_t["p_set"]
        for y in years
    }

    # Storage costs
    if storage_units:
        nodal_storage_costs = {
            y: opt_networks[y].storage_units_t["mu_energy_balance"]
            * opt_networks[y].storage_units_t["state_of_charge"]
            for y in years
        }
    else:
        nodal_storage_costs = {
            y: opt_networks[y].stores_t["mu_energy_balance"]
            * opt_networks[y].stores_t["e"]
            for y in years
        }
    # Fuel cell costs, only if storage units are not present
    if not storage_units:
        prod = {}
        shadow_price_fc = {}
        for y, n in opt_networks.items():
            fc_i = n.links.filter(like="H2 Fuel Cell", axis="rows").index
            prod[y] = n.links_t.p0.loc[:, fc_i]
            # Set to 0, whenever production is below 0.1 [MW].
            prod[y] = prod[y].where(prod[y] > 0.1, 0)
            shadow_price_fc[y] = -n.links_t.mu_upper.loc[:, fc_i]

        nodal_fc_costs = {y: prod[y] * shadow_price_fc[y] for y in years}
    if by_node:
        if storage_units:
            return nodal_costs, nodal_storage_costs
        else:
            return nodal_costs, nodal_storage_costs, nodal_fc_costs
    else:
        total_costs = {y: C.sum(axis="columns") for y, C in nodal_costs.items()}
        total_storage_costs = {
            y: C.sum(axis="columns") for y, C in nodal_storage_costs.items()
        }
        if storage_units:
            return total_costs, total_storage_costs
        else:
            total_fc_costs = {
                y: C.sum(axis="columns") for y, C in nodal_fc_costs.items()
            }
            return total_costs, total_storage_costs, total_fc_costs


def compute_anomalies_periods(
    r: gpd.GeoDataFrame,
    n: pypsa.Network,
    periods: pd.DataFrame,
    caps: pd.DataFrame,
    cfs: pd.DataFrame,
    means: pd.DataFrame,
    tech: str = "wind",
):
    """Compute the anomalies for the given periods and technology.
    Parameters:
    -----------
    r: gpd.GeoDataFrame
        GeoDataFrame with the regions.
    n: pypsa.Network
        PyPSA network.
    periods: pd.DataFrame
        DataFrame with the periods.
    caps: pd.DataFrame
        DataFrame with the capacities.
    cfs: pd.DataFrame
        DataFrame with the capacity factors.
    means: pd.DataFrame
        DataFrame with the means.
    tech: str
        Technology for which the anomalies are computed. Default is "wind".
    Returns:
    --------
    pd.DataFrame
        DataFrame with the anomalies for the given periods and technology.
    """
    anom_df = pd.DataFrame(index=r.index, columns=periods.index)
    for i, period in periods.iterrows():
        start, end = period.start, period.end
        net_year = get_net_year(start)
        shifted_start = (
            f"1942{str(start)[4:]}" if start.month < 7 else f"1941{str(start)[4:]}"
        )
        shifted_end = f"1942{str(end)[4:]}" if end.month < 7 else f"1941{str(end)[4:]}"

        usual_gen = (
            means.loc[shifted_start:shifted_end].mean(axis=0) * caps[str(net_year)]
        )
        event_gen = cfs.loc[start:end].mean(axis=0) * caps[str(net_year)]
        gen_anom = event_gen - usual_gen
        if tech not in ["load", "net_load", "price"]:
            gen_anom.index = n.generators.loc[gen_anom.index].bus.values
            gen_anom = gen_anom.groupby(gen_anom.index).sum().round(0)
        anom_df[i] = gen_anom
    return anom_df

def cost_acc_netw(
            n: pypsa.Network,
            start: str, 
            end: str,
            ):
    """Compute the cost accounting for a given network `n` over the period from `start` to `end`.
    Parameters:
    -----------
    n: PyPSA network
        The network for which the cost accounting is performed.
    start: str
        The start date of the period.
    end: str
        The end date of the period.
    
    Returns:
    --------
    dict
        A dictionary with the cost accounting values for different components of the network.
    """

    vals = {}
    price_nodes = n.buses.loc[n.buses.carrier == "AC"].index

    wind_generators = n.generators.loc[n.generators.carrier.str.contains("wind")].index
    solar_generators = n.generators.loc[n.generators.carrier.str.contains("solar")].index
    ror_generators = n.generators.loc[n.generators.carrier == "ror"].index
    renewable_generators = n.generators.loc[n.generators.carrier.isin(['solar-hsat', 'onwind', 'offwind-float',
       'solar', 'offwind-ac', 'offwind-dc', 'ror'])].index

    fc_i = n.links.loc[n.links.carrier == "H2 fuel cell"].index
    batt_i = n.links.loc[n.links.carrier == "battery charger"].index

    links_i = n.links.loc[n.links.carrier == "DC"].index

    # Congestion rent lines
    vals["lines"] = (((- n.lines_t.mu_upper + n.lines_t.mu_lower)*(n.lines.s_nom_opt)).loc[start:end].sum(axis=0).sum()/((- n.lines_t.mu_upper + n.lines_t.mu_lower)*(n.lines.s_nom_opt)).sum(axis=0).sum()).round(3)
    # Congestion rent links
    vals["links"] = ((- n.links_t.mu_upper + n.links_t.mu_lower) * n.links.p_nom_opt).loc[start:end, links_i].sum().sum() / ((- n.links_t.mu_upper + n.links_t.mu_lower) * n.links.p_nom_opt).loc[:, links_i].sum().sum()

    # Costs 
    vals["costs"] = (n.buses_t.marginal_price[price_nodes] * n.loads_t.p_set).loc[start:end].sum().sum()/(n.buses_t.marginal_price[price_nodes] * n.loads_t.p_set).sum().sum()

    # Renewables
    for generators, col in zip([wind_generators, solar_generators, ror_generators, renewable_generators], ["wind", "solar", "ror", "renew"]):
        gen = n.generators_t.p[generators]
        gen.columns = generators.map(n.generators.bus)
        vals[col] = (gen * n.buses_t.marginal_price[price_nodes]).loc[start:end].sum().sum()/(gen * n.buses_t.marginal_price[price_nodes]).sum().sum()
    
    # Storage
    for storage, col in zip([fc_i, batt_i], ["fc", "batt"]):
       vals[col] = (-n.links_t.p1[storage] * n.buses_t.marginal_price[price_nodes].values).loc[start:end].sum().sum()/(-n.links_t.p1[storage] * n.buses_t.marginal_price[price_nodes].values).sum().sum()  
    return vals



def cost_acc(opt_networks, years = None, periods = None):
    """
    Compute the cost accounting for the optimal networks over the given years or periods.

    Parameters:
    -----------
    opt_networks: dict
        Dictionary of PyPSA networks indexed by year.
    years: list, optional
        List of years for which the cost accounting is performed. If None, periods must be provided.
    periods: pd.DataFrame, optional
        DataFrame with the periods for which the cost accounting is performed. If None, years must be provided.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with the cost accounting values for different components of the networks, indexed by year or period.

    """
    cols = ["costs", "renew", "fc", "batt", "links", "lines", "wind", "solar", "ror"]

    if years is None and periods is None:
        raise ValueError("Either years or periods must be provided.")
    elif years is not None:
        df = pd.DataFrame(index = years, columns = cols).astype(float)
        for year, n in opt_networks.items():
            df.loc[year] = cost_acc_netw(n, f"{year}-10-01", f"{year+1}-03-31")
    else:
        df = pd.DataFrame(index = periods.index, columns = cols).astype(float)
        for i, period in periods.iterrows():
            n = opt_networks[get_net_year(period.start)]
            df.loc[i] = cost_acc_netw(n, period.start, period.end)
    return df


# Get network for a given date. Here, opt_networks[n] is defined over the period n-07-01 to (n+1)-06-30.
def get_net_year(date):
    """Get the year of the period from the date."""
    year = date.year
    if date.month < 7:
        year -= 1
    return year


def get_year_period(row):
    """Get the year of the period from the start date."""
    return get_net_year(row["start"])


def load_opt_networks(
    config_name: str,
    config_str: str = "base_s_90_elec_lc1.25_Co2L",
    load_networks: bool = True,
):
    """Load the configuration, scenario definition, years and optimal networks.

    Parameters:
    -----------
    config_name: str
        Name of the configuration file.
    config_str: str
        String defining the configuration options.
    load_networks: bool
        If True, load the optimal networks.

    Returns:
    --------
    config: dict
        Configuration dictionary loaded from the YAML file.
    scenario_def: dict
        Scenario definition dictionary loaded from the YAML file.
    years: list
        List of years for which the scenarios are defined.
    opt_networks: dict or None
        Dictionary of optimal networks indexed by year, or None if `load_networks` is False.
    """

    # Load the configuration
    with open(f"../pypsa-eur/config/{config_name}.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load scenario definition
    scenario_file = config["run"]["scenarios"]["file"]
    with open(f"../pypsa-eur/{scenario_file}", "r") as f:
        scenario_def = yaml.safe_load(f)

    # Load years
    scenario_names = list(scenario_def.keys())
    scenarios = {int(sn.split("_")[-1]): sn for sn in scenario_names}
    years = list(scenarios.keys())

    # Load optimal networks
    if load_networks:
        opt_networks = {
            year: pypsa.Network(
                f"../pypsa-eur/results/{config_name}/{scenario_name}/networks/{config_str}.nc"
            )
            for year, scenario_name in scenarios.items()
        }
    else:
        opt_networks = None

    return config, scenario_def, years, opt_networks


def load_periods(config: dict, clusters: str = None, ll: str = None, opts: str = None):
    """Load the system-defining events based on the configuration. The manual input is useful, if more than one scenario is being considered.

    Parameters:
    -----------
    config: dict
        PyPSA-Eur configuration.
    clusters: str
        Manual input for clusters.
    ll: str
        Manual input for transmission limits.
    opts: str
        Manual input for options.

    Returns:
    --------
    periods: pd.DataFrame
        DataFrame with the periods, indexed by their start time, with columns for start, end, and peak hour.
    """
    scen_name = config["difficult_periods"]["scen_name"]
    if clusters is None:
        clusters = config["scenario"]["clusters"][0]
    if ll is None:
        ll = config["scenario"]["ll"][0]
    if opts is None:
        opts = config["scenario"]["opts"][0]
    periods_name = f"sde_{scen_name}_{clusters}_elec_l{ll}_{opts}"

    periods = pd.read_csv(
        f"../pypsa-eur/results/periods/{periods_name}.csv",
        index_col=0,
        parse_dates=["start", "end", "peak_hour"],
    )

    for col in ["start", "end", "peak_hour"]:
        periods[col] = periods[col].dt.tz_localize(None)
    return periods


def load_sens_periods(sens_config, cost_thresholds):
    """Load the system-defining events based on the configuration.

    Parameters:
    -----------
    sens_config: dict
        Configuration for the sensitivity analysis.
    cost_thresholds: list
        List of cost thresholds to consider for the sensitivity analysis.

    Returns:
    --------
    sens_periods: dict
        Dictionary with cost thresholds as keys and DataFrames of periods as values."""
    file_names = {cost_threshold: {} for cost_threshold in cost_thresholds}
    sens_periods = {cost_threshold: {} for cost_threshold in cost_thresholds}
    for cost_threshold in cost_thresholds:
        dict_name = f"new_store_1941-2021_{cost_threshold}bn_12-336h"
        clusters = sens_config["scenario"]["clusters"]
        lls = sens_config["scenario"]["ll"]
        optss = sens_config["scenario"]["opts"]

        # Take the product of the lists of clusters, lls and optss
        for cluster in clusters:
            for ll in lls:
                for opts in optss:
                    periods_name = f"sde_{dict_name}_{cluster}_elec_l{ll}_{opts}"
                    file_names[cost_threshold][(cluster, ll, opts)] = periods_name
                    sens_periods[cost_threshold][(cluster, ll, opts)] = pd.read_csv(
                        f"../pypsa-eur/results/periods/{periods_name}.csv",
                        index_col=0,
                        parse_dates=["start", "end", "peak_hour"],
                    )

    return sens_periods


def optimal_costs(
    opt_networks: dict,
    techs: list = [
        "variable",
        "H2",
        "battery",
        "transmission-ac",
        "transmission-dc",
        "onwind",
        "offwind",
        "solar",
    ],
    pretty_names: dict = {
        "variable": "Variable costs",
        "transmission-ac": "AC  transmission",
        "transmission-dc": "DC transmission",
        "onwind": "Onshore wind",
        "offwind": "Offshore wind",
        "solar": "Solar",
        "H2": "Hydrogen",
        "battery": "Battery",
    },
    storage_units: bool = False,
):
    """Compute the optimal costs for the optimal networks. Some hard-coded features.

    Parameters:
    -----------
    opt_networks: dict
        Dictionary of PyPSA networks.
    techs: list
        List of technologies to consider.
    pretty_names: dict
        Dictionary of pretty names for the technologies.
    storage_units: bool
        If True, storage units are considered, otherwise stores.

    Returns:
    --------
    opt_objs: pd.DataFrame
        DataFrame with the optimal costs for the technologies, indexed by year.
    obj_totals: pd.DataFrame
        DataFrame with the total objective values for the optimal networks, indexed by year."""

    years = list(opt_networks.keys())
    opt_objs = pd.DataFrame(index=years, columns=techs)
    vres = [
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "solar",
        "offwind-float",
        "solar-hsat",
    ]

    for y, n in opt_networks.items():
        # Generators
        g_inv = n.generators["p_nom_opt"] * n.generators["capital_cost"]
        for g in vres:
            i = n.generators.loc[n.generators.carrier == g].index
            opt_objs.loc[y, g] = g_inv.loc[i].sum()
        # Transmission:
        opt_objs.loc[y, "transmission-ac"] = (
            (n.lines.s_nom_opt - n.lines.s_nom) * n.lines.capital_cost
        ).sum()
        opt_objs.loc[y, "transmission-dc"] = (
            (n.links.loc[n.links.carrier.str.contains("DC")].p_nom_opt - n.links.p_nom)
            * n.links.loc[n.links.carrier.str.contains("DC")].capital_cost
        ).sum()
        # Storage
        if storage_units:
            s_inv = n.storage_units["p_nom_opt"] * n.storage_units["capital_cost"]
            for s in ["H2", "battery"]:
                i = n.storage_units.loc[n.storage_units.carrier == s].index
                opt_objs.loc[y, s] = s_inv.loc[i].sum()
        else:
            # Stores instead of storage units
            s_inv = n.stores["e_nom_opt"] * n.stores["capital_cost"]
            for s in ["H2", "battery"]:
                i = n.stores.loc[n.stores.carrier == s].index
                opt_objs.loc[y, s] = s_inv.loc[i].sum()
            # Add charging and discharging capacities.
            s_c_inv = n.links["p_nom_opt"] * n.links["capital_cost"]
            for s in ["H2", "battery"]:
                i = n.links.loc[n.links.carrier.str.contains(s)].index
                opt_objs.loc[y, s] += s_c_inv.loc[i].sum()
        # Variable:
        opt_objs.loc[y, "variable"] = annual_variable_cost(n)

    opt_objs["offwind"] = (
        opt_objs["offwind-ac"] + opt_objs["offwind-dc"] + opt_objs["offwind-float"]
    )
    opt_objs["solar"] = opt_objs["solar"] + opt_objs["solar-hsat"]
    del opt_objs["offwind-ac"]
    del opt_objs["offwind-dc"]
    del opt_objs["offwind-float"]
    del opt_objs["solar-hsat"]

    # Also compile the total objective values as a sanity check; n.objective
    # should be equal to the sum of the individual objective values.
    obj_totals = pd.DataFrame(index=years, columns=["total"])
    obj_totals["total"] = [n.objective for n in opt_networks.values()]
    opt_objs = opt_objs.rename(pretty_names, axis="columns")
    # Change the index to the form "80/81"
    opt_objs.index = [f"{str(y)[-2:]}/{str(y + 1)[-2:]}" for y in opt_objs.index]
    return opt_objs, obj_totals


## FLEXIBILITY CALCULATIONS


# Flexibility in use (system-wide)
def calculate_transmission_in_use(n):
    """Calculate the in-use ratios and capacities for transmission.

    Parameters:
    -----------
    n: PyPSA network.

    Returns:
    --------
    transmission_in_use: pd.DataFrame
        DataFrame with the in-use ratios for transmission links and lines, indexed by snapshot.
    transmission_p_nom: pd.Series
        Series with the nominal capacities for transmission links and lines.
    """
    # Transmission links
    dc_i = n.links[n.links.carrier == "DC"].index
    dc_p_nom = n.links.loc[dc_i].p_nom_opt
    dc_in_use = ((n.links_t.p0.loc[:, dc_i].abs()) / (0.99 * dc_p_nom)).clip(upper=1)

    # Transmission lines
    ac_i = n.lines.index
    ac_s_nom = n.lines.loc[ac_i, "s_nom_opt"]
    ac_in_use = (
        (n.lines_t.p0.loc[:, ac_i].abs())
        / (0.99 * ac_s_nom * n.lines.loc[ac_i, "s_max_pu"])
    ).clip(upper=1)

    # Combine DC and AC transmission data
    transmission_in_use = pd.concat([dc_in_use, ac_in_use], axis=1)
    transmission_p_nom = pd.concat([dc_p_nom, ac_s_nom], axis=0)

    return transmission_in_use, transmission_p_nom


def calculate_dispatchable_in_use(n):
    """Calculate the in-use ratios and capacities for dispatchable technologies.

    Parameters:
    -----------
    n: PyPSA network.

    Returns:
    --------
    dispatchable_in_use: pd.DataFrame
        DataFrame with the in-use ratios for dispatchable technologies, indexed by snapshot.
    dispatchable_p_nom: pd.Series
        Series with the nominal capacities for dispatchable technologies.
    """
    # Dispatchable technologies
    biomass_i = n.generators[n.generators.carrier == "biomass"].index
    nuclear_i = n.generators[n.generators.carrier == "nuclear"].index
    ror_i = n.generators[n.generators.carrier == "ror"].index

    biomass_p_nom = n.generators.loc[biomass_i].p_nom_opt
    nuclear_p_nom = n.generators.loc[nuclear_i].p_nom_opt
    ror_p_nom = n.generators.loc[ror_i].p_nom_opt
    dispatchable_p_nom = pd.concat([biomass_p_nom, nuclear_p_nom, ror_p_nom], axis=0)

    # Calculate in-use ratios for dispatchable technologies
    biomass_in_use = (n.generators_t.p.loc[:, biomass_i] / (0.99 * biomass_p_nom)).clip(
        upper=1
    )
    nuclear_in_use = (n.generators_t.p.loc[:, nuclear_i] / (0.99 * nuclear_p_nom)).clip(
        upper=1
    )
    ror_in_use = (
        n.generators_t.p.loc[:, ror_i]
        / ror_p_nom
        * n.generators_t.p_max_pu.loc[:, ror_i]
    )

    dispatchable_in_use = pd.concat(
        [biomass_in_use, nuclear_in_use, ror_in_use], axis=1
    )

    return dispatchable_in_use, dispatchable_p_nom


def calculate_storage_discharge_in_use(n):
    """Calculate the in-use ratios and capacities for storage discharge technologies.

    Parameters:
    -----------
    n: PyPSA network.

    Returns:
    --------
    storage_discharge_in_use: pd.DataFrame
        DataFrame with the in-use ratios for storage discharge technologies, indexed by snapshot.
    storage_p_nom: pd.Series
        Series with the nominal capacities for storage discharge technologies.
    """
    # Storage discharge
    fuel_cells_i = n.links[n.links.carrier == "H2 fuel cell"].index
    battery_i = n.links[n.links.carrier == "battery discharger"].index
    phs_i = n.storage_units[n.storage_units.carrier == "PHS"].index
    hydro_i = n.storage_units[n.storage_units.carrier == "hydro"].index

    fuel_cells_p_nom = (
        n.links.loc[fuel_cells_i].p_nom_opt * n.links.loc[fuel_cells_i].efficiency
    )
    battery_p_nom = n.links.loc[battery_i].p_nom_opt * n.links.loc[battery_i].efficiency
    phs_p_nom = n.storage_units.loc[phs_i].p_nom_opt
    hydro_p_nom = n.storage_units.loc[hydro_i].p_nom_opt
    storage_p_nom = pd.concat(
        [fuel_cells_p_nom, battery_p_nom, phs_p_nom, hydro_p_nom], axis=0
    )

    # Calculate in-use ratios for storage discharge
    fuel_cells_in_use = (
        n.links_t.p1.loc[:, fuel_cells_i].abs() / (0.99 * fuel_cells_p_nom)
    ).clip(upper=1)
    battery_in_use = (
        n.links_t.p1.loc[:, battery_i].abs() / (0.99 * battery_p_nom)
    ).clip(upper=1)
    phs_in_use = (n.storage_units_t.p.loc[:, phs_i] / (0.99 * phs_p_nom)).clip(upper=1)
    hydro_in_use = (n.storage_units_t.p.loc[:, hydro_i] / (0.99 * hydro_p_nom)).clip(
        upper=1
    )

    storage_discharge_in_use = pd.concat(
        [fuel_cells_in_use, battery_in_use, phs_in_use, hydro_in_use], axis=1
    )

    return storage_discharge_in_use, storage_p_nom


def clean_incidence_matrix(
    periods: pd.DataFrame,
    sens_periods: dict,
    scenario_list: list,
):
    """Create a cleaned incidence matrix for the sensitivity periods.

    Parameters:
    -----------
    periods: pd.DataFrame
        DataFrame with the periods.
    sens_periods: dict
        Dictionary with the sensitivity periods, where keys are scenario names and values scenario names.
    scenario_list: list
        List of scenarios to consider in the incidence matrix.

    Returns:
    --------
    matrix: np.ndarray
        Incidence matrix where rows correspond to scenarios and columns to periods.
    """
    matrix = np.zeros((len(scenario_list), len(periods)))
    for costs in reversed(sens_periods.keys()):
        for i, scenar in enumerate(scenario_list):
            if scenar == (90, "c1.25", "Co2L0.0"):
                matrix[i, :] = 4
            else:
                alt_periods = sens_periods[costs][scenar]
                if len(alt_periods) == 0:
                    matrix[i, :] = 0
                else:
                    for j, ind in enumerate(periods.index):
                        period = periods.loc[ind]
                        start = period.start.tz_localize(tz="UTC")
                        end = period.end.tz_localize(tz="UTC")
                        time_slice = pd.date_range(start, end, freq="h")
                        for k, alt_period in alt_periods.iterrows():
                            alt_time_slice = pd.date_range(
                                alt_period.start, alt_period.end, freq="h"
                            )
                            if (
                                len(set(time_slice).intersection(set(alt_time_slice)))
                                > 0
                            ):
                                matrix[i, j] += 1
                                break
    return matrix


def coarse_system_flex(m):
    """Calculate the coarse system flexibility for transmission, dispatchable and storage technologies.

    Parameters:
    -----------
    m: PyPSA network.

    Returns:
    --------
    system_flex_coarse: pd.DataFrame
        DataFrame with the coarse system flexibility for transmission, dispatchable and storage discharge technologies, indexed by snapshot.

    """
    transmission_in_use, transmission_p_nom = calculate_transmission_in_use(m)
    dispatchable_in_use, dispatchable_p_nom = calculate_dispatchable_in_use(m)
    storage_discharge_in_use, storage_p_nom = calculate_storage_discharge_in_use(m)

    system_flex_coarse = pd.DataFrame(
        columns=["transmission", "dispatchable", "storage_discharge"],
        index=transmission_in_use.index,
    )

    system_flex_coarse["transmission"] = (transmission_in_use * transmission_p_nom).sum(
        axis=1
    ) / transmission_p_nom.sum()
    system_flex_coarse["dispatchable"] = (dispatchable_in_use * dispatchable_p_nom).sum(
        axis=1
    ) / dispatchable_p_nom.sum()
    system_flex_coarse["storage_discharge"] = (
        storage_discharge_in_use * storage_p_nom
    ).sum(axis=1) / storage_p_nom.sum()

    return system_flex_coarse


def detailed_system_flex(m):
    """Calculate the detailed system flexibility for the different technologies.

    Parameters:
    -----------
    m: PyPSA network.

    Returns:
    --------
    system_flex_detailed: pd.DataFrame
        DataFrame with the detailed system flexibility for the different technologies, indexed by snapshot.
    """
    transmission_in_use, transmission_p_nom = calculate_transmission_in_use(m)
    dispatchable_in_use, dispatchable_p_nom = calculate_dispatchable_in_use(m)
    storage_discharge_in_use, storage_p_nom = calculate_storage_discharge_in_use(m)

    system_flex_detailed = pd.DataFrame(
        columns=[
            "DC",
            "AC",
            "biomass",
            "nuclear",
            "ror",
            "fuel_cells",
            "battery",
            "phs",
            "hydro",
        ],
        index=transmission_in_use.index,
    )

    dc_i = m.links[m.links.carrier == "DC"].index
    ac_i = m.lines.index
    biomass_i = m.generators[m.generators.carrier == "biomass"].index
    nuclear_i = m.generators[m.generators.carrier == "nuclear"].index
    ror_i = m.generators[m.generators.carrier == "ror"].index
    fuel_cells_i = m.links[m.links.carrier == "H2 fuel cell"].index
    battery_i = m.links[m.links.carrier == "battery discharger"].index
    phs_i = m.storage_units[m.storage_units.carrier == "PHS"].index
    hydro_i = m.storage_units[m.storage_units.carrier == "hydro"].index

    biomass_p_nom = m.generators.loc[biomass_i].p_nom_opt
    nuclear_p_nom = m.generators.loc[nuclear_i].p_nom_opt
    ror_p_nom = m.generators.loc[ror_i].p_nom_opt
    fuel_cells_p_nom = m.links.loc[fuel_cells_i].p_nom_opt
    battery_p_nom = m.links.loc[battery_i].p_nom_opt
    phs_p_nom = m.storage_units.loc[phs_i].p_nom_opt
    hydro_p_nom = m.storage_units.loc[hydro_i].p_nom_opt

    system_flex_detailed["DC"] = (
        transmission_in_use[dc_i] * transmission_p_nom[dc_i]
    ).sum(axis=1) / transmission_p_nom[dc_i].sum()
    system_flex_detailed["AC"] = (
        transmission_in_use[ac_i] * transmission_p_nom[ac_i]
    ).sum(axis=1) / transmission_p_nom[ac_i].sum()
    system_flex_detailed["biomass"] = (
        dispatchable_in_use[biomass_i] * biomass_p_nom
    ).sum(axis=1) / biomass_p_nom.sum()
    system_flex_detailed["nuclear"] = (
        dispatchable_in_use[nuclear_i] * nuclear_p_nom
    ).sum(axis=1) / nuclear_p_nom.sum()
    system_flex_detailed["ror"] = (dispatchable_in_use[ror_i] * ror_p_nom).sum(
        axis=1
    ) / ror_p_nom.sum()
    system_flex_detailed["fuel_cells"] = (
        storage_discharge_in_use[fuel_cells_i] * fuel_cells_p_nom
    ).sum(axis=1) / fuel_cells_p_nom.sum()
    system_flex_detailed["battery"] = (
        storage_discharge_in_use[battery_i] * battery_p_nom
    ).sum(axis=1) / battery_p_nom.sum()
    system_flex_detailed["phs"] = (storage_discharge_in_use[phs_i] * phs_p_nom).sum(
        axis=1
    ) / phs_p_nom.sum()
    system_flex_detailed["hydro"] = (
        storage_discharge_in_use[hydro_i] * hydro_p_nom
    ).sum(axis=1) / hydro_p_nom.sum()

    return system_flex_detailed


def nodal_flexibility(
    opt_networks: dict,
    nodes: list,
    tech: list = [
        "DC",
        "AC",
        "biomass",
        "nuclear",
        "ror",
        "H2 fuel cell",
        "battery discharger",
        "PHS",
        "hydro",
    ],
):
    """Calculate the nodal flexibility for the different technologies.

    Parameters:
    -----------
    opt_networks: dict
        Dictionary of PyPSA networks.
    nodes: list
        List of nodes.
    tech: list
        List of technologies.

    Returns:
    --------
    nodal_flex_p: dict
        Dictionary with nodal flexibility capacities for each technology, indexed by node and year.
    nodal_flex_u: dict
        Dictionary with nodal flexibility in-use ratios for each technology, indexed by node and year.
    """
    nodal_flex_u = {}
    nodal_flex_p = {}

    transmission_in_use = {}
    transmission_p_nom = {}
    dispatchable_in_use = {}
    dispatchable_p_nom = {}
    storage_discharge_in_use = {}
    storage_p_nom = {}

    for y, n in opt_networks.items():
        transmission_in_use[y], transmission_p_nom[y] = calculate_transmission_in_use(n)
        dispatchable_in_use[y], dispatchable_p_nom[y] = calculate_dispatchable_in_use(n)
        storage_discharge_in_use[y], storage_p_nom[y] = (
            calculate_storage_discharge_in_use(n)
        )

    for node in nodes:
        flex_p = pd.DataFrame(columns=tech)
        list_flex_u = []
        for y, n in opt_networks.items():
            flex_u = pd.DataFrame(columns=tech, index=n.snapshots)
            for t in tech:
                # Transmission
                if t == "DC":
                    i = n.links[
                        (n.links.carrier == "DC")
                        & ((n.links.bus0 == node) | (n.links.bus1 == node))
                    ].index
                    flex_p.loc[y, t] = n.links.loc[i, "p_nom_opt"].sum()
                    flex_u.loc[:, t] = (
                        (
                            transmission_in_use[y].loc[:, i]
                            * n.links.loc[i, "p_nom_opt"]
                        ).sum(axis="columns")
                    ) / flex_p.loc[y, t]
                elif t == "AC":
                    i = n.lines[((n.lines.bus0 == node) | (n.lines.bus1 == node))].index
                    flex_p.loc[y, t] = (n.lines.loc[i, "s_nom_opt"]).sum()
                    flex_u.loc[:, t] = (
                        (
                            transmission_in_use[y].loc[:, i]
                            * n.lines.loc[i, "s_nom_opt"]
                        ).sum(axis="columns")
                    ) / flex_p.loc[y, t]
                # Dispatch
                elif t in ["biomass", "nuclear", "ror"]:
                    i = n.generators[
                        (n.generators.bus == node) & (n.generators.carrier == t)
                    ].index
                    flex_p.loc[y, t] = n.generators.loc[i, "p_nom_opt"].sum()
                    flex_u.loc[:, t] = (
                        (
                            dispatchable_in_use[y].loc[:, i]
                            * n.generators.loc[i, "p_nom_opt"]
                        ).sum(axis="columns")
                    ) / flex_p.loc[y, t]
                # Storage
                else:
                    if t in ["H2 fuel cell", "battery discharger"]:
                        i = n.links[
                            (n.links.carrier == t) & (n.links.bus1 == node)
                        ].index
                        flex_p.loc[y, t] = (
                            n.links.loc[i, "p_nom_opt"] * n.links.loc[i, "efficiency"]
                        ).sum()
                        flex_u.loc[:, t] = (
                            (
                                storage_discharge_in_use[y].loc[:, i]
                                * n.links.loc[i, "p_nom_opt"]
                                * n.links.loc[i, "efficiency"]
                            ).sum(axis="columns")
                        ) / flex_p.loc[y, t]
                    elif t in ["PHS", "hydro"]:
                        i = n.storage_units[
                            (n.storage_units.carrier == t)
                            & (n.storage_units.bus == node)
                        ].index
                        flex_p.loc[y, t] = n.storage_units.loc[i, "p_nom_opt"].sum()
                        flex_u.loc[:, t] = (
                            (
                                storage_discharge_in_use[y].loc[:, i]
                                * n.storage_units.loc[i, "p_nom_opt"]
                            ).sum(axis="columns")
                        ) / flex_p.loc[y, t]
                    else:
                        print(f"{t} not implemented.")
                        continue
                flex_u = flex_u.astype(float).round(2)
            list_flex_u.append(flex_u)
        nodal_flex_p[node] = flex_p
        nodal_flex_u[node] = pd.concat(list_flex_u)
    return nodal_flex_p, nodal_flex_u


# NOTE: For ror, we only have limited availability in the different time steps, so the flexibility availability in capacities can be slightly misleading.




## PLOTTING


def plot_affected_areas(
    config_name: str,
    period: pd.Series,
    event_nr: int,
    hulls: list[ConvexHull],
    techs: list[str],
    pretty_names: list[str],
    colours: list[str],
    fill_df: pd.DataFrame,
    fill_tech: str,
    fill_norm: mpl.colors.Normalize,
    fill_cmap: str,
    regions: gpd.GeoDataFrame,
    n: pypsa.Network,
    projection: ccrs.Projection,
    ax: plt.Axes = None,
    save: bool = False,
) -> list:
    """Plot affected areas on a map based on pre-computed hulls.

    Parameters:
    -----------
    config_name: str
        Name of the configuration
    period: pd.Series
        Period data for the event
    event_nr: int
        Event number
    hulls: list[ConvexHull]
        List of pre-computed hulls for each technology
    techs: list[str]
        List of technology names
    pretty_names: list[str]
        Pretty names for technologies for the legend
    colours: list[str]
        Colors to use for the hulls
    fill_df: pd.DataFrame
        Data to use for filling in the regions
    fill_tech: str
        Technology to use for coloring regions
    fill_norm: mpl.colors.Normalize
        Color normalization for the fill
    fill_cmap: str
        Colormap for the fill
    regions: gpd.GeoDataFrame
        Regions to plot
    n: pypsa.Network
        Network to plot
    projection: ccrs.Projection
        Projection to use
    ax: plt.Axes, optional
        Axes to plot on
    save: bool, default False
        Whether to save the plot
    """
    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 10), subplot_kw={"projection": projection}
        )
    n.plot(
        ax=ax,
        bus_sizes=0,
        bus_colors="black",
        line_widths=0,
        link_widths=0,
        link_colors="black",
        line_colors="black",
        color_geomap=True,
    )

    start = period.start
    end = period.end

    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    r[fill_tech] = fill_df.loc[start:end].mean()

    r.plot(
        ax=ax,
        column=fill_tech,
        cmap=fill_cmap,
        norm=fill_norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    ax.set_title("Fuel cell usage", fontsize=8)

    # Add cbar.
    sm = plt.cm.ScalarMappable(cmap=fill_cmap, norm=fill_norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, orientation="vertical", pad=0.05, aspect=25, shrink=0.75
    )

    ticks = [0, 0.25, 0.5, 0.75, 1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0%}" for t in ticks], fontsize=6)

    legend_elements = []
    hatches = [None, None, None, "x", "/", "o", ".", "*", "O", "-"]

    for hull, tech, colour, pretty_name, hatch in zip(
        hulls, techs, colours, pretty_names, hatches
    ):
        # For now, only edges, no filling.
        hull_transformed = projection.transform_points(
            ccrs.PlateCarree(),
            hull.points[hull.vertices][:, 0],
            hull.points[hull.vertices][:, 1],
        )
        patch = Polygon(
            xy=hull_transformed[:, :2],
            closed=True,
            ec=colour,
            fill=False,
            lw=1,
            zorder=2,
        )
        legend_elements.append(Patch(ec=colour, fill=False, lw=1, label=pretty_name))
        ax.add_patch(patch)
    return legend_elements


def plot_anomalies(
    tech: str,
    n: pypsa.Network,
    cmap: mpl.colors.ListedColormap,
    norm: mpl.colors.BoundaryNorm,
    regions: gpd.GeoDataFrame,
    periods: pd.DataFrame,
    caps: pd.DataFrame,
    cfs: pd.DataFrame,
    means: pd.DataFrame,
    ax: plt.Axes,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    cbar: bool = True,
    cbar_label: str = None,
    offshore_regions: gpd.GeoDataFrame = None,
    scaling_factor: float = 1,
):
    """Plot the anomalies for the given technology and network.
    Parameters:
    -----------
    tech: str
        Technology for which the anomalies are plotted.
    n: pypsa.Network
        PyPSA network.
    cmap: mpl.colors.ListedColormap
        Colormap for the anomalies.
    norm: mpl.colors.BoundaryNorm
        Normalization for the colormap.
    regions: gpd.GeoDataFrame
        GeoDataFrame with the regions.
    periods: pd.DataFrame
        DataFrame with the system-defining events.
    caps: pd.DataFrame
        DataFrame with the capacities.
    cfs: pd.DataFrame
        DataFrame with the capacity factors.
    means: pd.DataFrame
        DataFrame with the means.
    ax: plt.Axes
        Axes to plot on.
    projection: ccrs.Projection
        Projection for the plot.
    cbar: bool
        If True, add a colorbar to the plot.
    cbar_label: str
        Label for the colorbar.
    offshore_regions: gpd.GeoDataFrame
        GeoDataFrame with the offshore regions.
    scaling_factor: float
        Scaling factor for the anomalies.
    """
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": projection},
            figsize=(6 * cm, 6 * cm),
        )
    if tech == "wind":
        if offshore_regions is None:
            print("No offshore regions provided, using onshore regions.")
            caps = caps.filter(like="onwind", axis=0)
            cfs = cfs.filter(like="onwind", axis=1)
            means = means.filter(like="onwind", axis=1)

    n.plot(ax=ax, bus_sizes=0, line_widths=0, link_widths=0)

    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    r["mean_anom"] = compute_anomalies_periods(
        r,
        n,
        periods,
        caps,
        cfs,
        means,
        tech,
    ).mean(axis=1)
    r["mean_anom"] *= scaling_factor

    r.plot(
        ax=ax,
        column="mean_anom",
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    if offshore_regions is not None:
        off_r = offshore_regions.set_index("name")
        off_r["x"], off_r["y"] = n.buses.x, n.buses.y
        off_r = gpd.geodataframe.GeoDataFrame(off_r, crs="EPSG:4326")
        off_r = off_r.to_crs(projection.proj4_init)

        off_caps = caps.filter(like="offwind", axis=0)
        off_cfs = cfs.filter(like="offwind", axis=1)
        off_means = means.filter(like="offwind", axis=1)

        off_r["mean_anom"] = compute_anomalies_periods(
            off_r,
            n,
            periods,
            off_caps,
            off_cfs,
            off_means,
        ).mean(axis=1)
        off_r["mean_anom"] *= scaling_factor
        off_r.plot(
            ax=ax,
            column="mean_anom",
            cmap=cmap,
            norm=norm,
            alpha=0.6,
            linewidth=0,
            zorder=1,
        )
    if cbar:
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.01,
            aspect=20,
            shrink=0.9,
            fraction=0.05,
        )
        cbar.set_label(f"{cbar_label}", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    if offshore_regions is not None:
        return r, off_r
    return r


def plot_clean_incidence_matrix(
    m: np.ndarray,
    ylabels: list,
    cmap: mpl.colors.ListedColormap = mpl.colors.ListedColormap(
        ["white", "mistyrose", "tomato", "maroon", "cornflowerblue"]
    ),
    ax: plt.Axes = None,
):
    """Plot a clean incidence matrix with hatches for zero values.

    Parameters:
    -----------
    m: np.ndarray
        The incidence matrix to plot.
    ylabels: list
        Labels for the y-axis.
    cmap: mpl.colors.ListedColormap
        Colormap to use for the matrix.
    ax: plt.Axes
        Axes to plot on. If None, a new figure and axes are created.
    """

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=4)
    # Mark all values that are 0 with a hatch.
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] == 0:
                ax.text(j, i, "X", ha="center", va="center", color="black")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    return im


def plot_cluster_anomaly(
    flex_anomaly: pd.DataFrame,
    system_anomaly: pd.DataFrame,
    periods: pd.DataFrame,
    event_nr: int,
    tech: list = [
        "biomass",
        "nuclear",
        "H2 fuel cell",
        "battery discharger",
        "PHS",
        "hydro",
    ],
    tech_colours: list = [
        "#baa741",
        "#ff8c00",
        "#c251ae",
        "#ace37f",
        "#51dbcc",
        "#298c81",
    ],
    plot_all_system: bool = True,
    resampled: str = "1D",
    ax=None,
):
    """Plot the flexibility and system anomalies for the all clusters.

    Parameters:
    -----------
    flex_anomaly: pd.DataFrame
        Flexibility anomalies per technology.
    system_anomaly: pd.DataFrame
        System anomalies (net load, load, wind, solar).
    periods: pd.DataFrame
        System-defining events.
    tech: list
        List of technologies.
    tech_colours: list
        List of colours for the technologies.
    plot_all_system: bool
        If True, plot all system anomalies.
    resampled: str
        Resampling frequency.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6 * cm, 4 * cm))
    start = periods.loc[event_nr, "start"]
    end = periods.loc[event_nr, "end"]

    # Plot stack plot of flexibility
    p = (
        flex_anomaly.loc[start:end, tech].astype(float).resample(resampled).mean() / 1e3
    )  # in GW
    p_neg = p.clip(upper=0)
    p_pos = p.clip(lower=0)
    ax.stackplot(p_pos.index, p_pos.T, colors=tech_colours, labels=p_pos.columns)
    ax.stackplot(p_neg.index, p_neg.T, colors=tech_colours)
    # Plot net load anomaly.
    net_load = (
        system_anomaly.loc[start:end, "Net load anomaly"].resample(resampled).mean()
        / 1e3
    )
    ax.plot(
        net_load.index,
        net_load,
        color="black",
        lw=1,
        ls="--",
        label="Net load anomaly",
    )
    if plot_all_system:
        load = (
            system_anomaly.loc[start:end, "Load anomaly"].resample(resampled).mean()
            / 1e3
        )
        wind = (
            system_anomaly.loc[start:end, "Wind anomaly"].resample(resampled).mean()
            / 1e3
        )
        solar = (
            system_anomaly.loc[start:end, "Solar anomaly"].resample(resampled).mean()
            / 1e3
        )
        ax.plot(load.index, load, color="red", lw=1, ls=":", label="Load anomaly")
        ax.plot(wind.index, wind, color="blue", lw=1, ls=":", label="Wind anomaly")
        ax.plot(
            solar.index,
            solar,
            color="grey",
            lw=1,
            ls=":",
            label="Solar anomaly",
        )
        ax.set_ylabel("Anomaly [GW]")
        ax.set_xlabel(f"{start.date()} - {end.date()}", fontsize=8)
        # Set x-ticks to only show day and month
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        # Change font size of x tick labels.
        ax.tick_params(axis="x", labelsize=8, rotation=90)
    return ax


def plot_cluster_anomalies(
    flex_anomaly: pd.DataFrame,
    system_anomaly: pd.DataFrame,
    periods: pd.DataFrame,
    cluster_nr: int,
    clustered_vals: pd.DataFrame,
    tech: list = [
        "biomass",
        "nuclear",
        "H2 fuel cell",
        "battery discharger",
        "PHS",
        "hydro",
    ],
    tech_colours: list = [
        "#baa741",
        "#ff8c00",
        "#c251ae",
        "#ace37f",
        "#51dbcc",
        "#298c81",
    ],
    plot_all_system: bool = True,
    resampled: str = "1D",
    save_fig: bool = False,
    path_str: str = None,
    cluster_names: list = None,
):
    """Plot the flexibility and system anomalies for the different clusters.

    Parameters:
    -----------
    flex_anomaly: pd.DataFrame
        Flexibility anomalies per technology.
    system_anomaly: pd.DataFrame
        System anomalies (net load, load, wind, solar).
    periods: pd.DataFrame
        System-defining events.
    cluster_nr: int
        Number of clusters.
    clustered_vals: pd.DataFrame
        Clustered values with stats for each SDE.
    tech: list
        List of technologies.
    tech_colours: list
        List of colours for the technologies.
    plot_all_system: bool
        If True, plot all system anomalies.
    resampled: str
        Resampling frequency.
    save_fig: bool
        If True, save the figure.
    path_str: str
        Path to save the figure.
    cluster_names: list
        List of cluster names.
    """
    for cluster in range(cluster_nr):
        nb_plots = len(clustered_vals.loc[clustered_vals["cluster"] == cluster])
        nb_rows = nb_plots // 4 if nb_plots % 4 == 0 else nb_plots // 4 + 1

        fig, axs = plt.subplots(
            nb_rows,
            4,
            figsize=(30 * cm, nb_rows * 7 * cm),
            sharey=True,
            gridspec_kw={"hspace": 0.6},
        )
        if len(cluster_names) == cluster_nr:
            fig.suptitle(f"{cluster_names[cluster]}", fontsize=16)
        else:
            fig.suptitle(f"Cluster {cluster}", fontsize=16)

        for i, event_nr in enumerate(
            clustered_vals[clustered_vals["cluster"] == cluster].index
        ):
            ax = axs.flatten()[i]
            start = periods.loc[event_nr, "start"]
            end = periods.loc[event_nr, "end"]

            # Plot stack plot of flexibility.
            p = (
                flex_anomaly.loc[start:end, tech]
                .astype(float)
                .resample(resampled)
                .mean()
                / 1e3
            )  # in GW
            p_neg = p.clip(upper=0)
            p_pos = p.clip(lower=0)
            ax.stackplot(
                p_pos.index, p_pos.T, colors=tech_colours, labels=p_pos.columns
            )
            ax.stackplot(p_neg.index, p_neg.T, colors=tech_colours)
            # Plot net load anomaly.
            net_load = (
                system_anomaly.loc[start:end, "Net load anomaly"]
                .resample(resampled)
                .mean()
                / 1e3
            )
            ax.plot(
                net_load.index,
                net_load,
                color="black",
                lw=1,
                ls="--",
                label="Net load anomaly",
            )
            if plot_all_system:
                load = (
                    system_anomaly.loc[start:end, "Load anomaly"]
                    .resample(resampled)
                    .mean()
                    / 1e3
                )
                wind = (
                    system_anomaly.loc[start:end, "Wind anomaly"]
                    .resample(resampled)
                    .mean()
                    / 1e3
                )
                solar = (
                    system_anomaly.loc[start:end, "Solar anomaly"]
                    .resample(resampled)
                    .mean()
                    / 1e3
                )
                ax.plot(
                    load.index, load, color="red", lw=1, ls=":", label="Load anomaly"
                )
                ax.plot(
                    wind.index, wind, color="blue", lw=1, ls=":", label="Wind anomaly"
                )
                ax.plot(
                    solar.index,
                    solar,
                    color="grey",
                    lw=1,
                    ls=":",
                    label="Solar anomaly",
                )
            ax.set_title(f"Event {event_nr}")
            ax.set_ylabel("Flexibility/Anomaly [GW]")
            ax.set_xlabel(f"{start.date()} - {end.date()}", fontsize=8)
            # Set x-ticks to only show day and month
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            # Change font size of x tick labels.
            ax.tick_params(axis="x", labelsize=8, rotation=90)
        # Add legend to the last row between the 2nd and 3rd plot.
        labels, handles = ax.get_legend_handles_labels()
        axs.flatten()[-3].legend(
            labels,
            handles,
            loc="center",
            bbox_to_anchor=(1, -0.6),
            frameon=False,
            ncols=4,
        )
        # Hide empty plots.
        for ax in axs.flatten()[nb_plots:]:
            ax.axis("off")
        if save_fig:
            if path_str is None:
                print("Please specify a path to save the figure.")
            else:
                plt.savefig(f"{path_str}_cluster_{cluster}.pdf", bbox_inches="tight")
        else:
            plt.show()


def plot_duals(
    periods: pd.DataFrame,
    years: list,
    left_vals: dict,
    right_vals: dict,
    left_norm,
    right_norm,
    left_cmap: str = "Blues",
    right_cmap: str = "Oranges",
    left_str: str = "Fuel cell costs",
    right_str: str = "Electricity costs",
    left_ticks: list = [1, 10, 100],
    right_ticks: list = [1, 10, 100, 1000, 10000],
    left_scaling: float = 1e-6,
    right_scaling: float = 1e-6,
    save_fig: bool = False,
    path_str: str = None,
    alt_periods: Optional[pd.DataFrame] = None,
):
    """Plot the dual values for the different years and possibly different duals where system-defining events (or other filtered) periods are identified.

    Parameters:
    -----------
    periods: pd.DataFrame
        System-defining events.
    years: list
        List of years.
    left_vals: dict
        Dictionary of left dual values, e.g. total_costs.
    right_vals: dict
        Dictionary of right dual values, e.g. total_costs.
    left_norm: mpl.colors.Normalize
        Normalization for the left dual values.
    right_norm: mpl.colors.Normalize
        Normalization for the right dual values.
    left_cmap: str
        Colormap for the left dual values.
    right_cmap: str
        Colormap for the right dual values.
    left_str: str
        String for the left dual values.
    right_str: str
        String for the right dual values.
    left_ticks: list
        Ticks for the left dual values.
    right_ticks: list
        Ticks for the right dual values.
    left_scaling: float
        Scaling factor for the left dual values.
    right_scaling: float
        Scaling factor for the right dual values.
    save_fig: bool
        If True, save the figure.
    path_str: str
        Path to save the figure.
    alt_periods: pd.DataFrame
        Alternative system-defining events to be marked.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18 * cm, 18 * cm))

    dates = mdates.date2num(
        pd.date_range("2014-10-01", "2015-03-31", freq="D").to_pydatetime()
    )

    for ax, year_selection in zip(
        axs, [years[: (len(years) // 2)], years[(len(years) // 2) :]]
    ):
        for y in year_selection:
            # Resample to daily resolution.
            C = left_vals[y].resample("D").mean()
            # Drop leap days from C
            C = C.loc[(C.index.month != 2) | (C.index.day != 29)]
            # Select only the period from October to March inclusive.
            C = C.loc[(C.index.month >= 10) | (C.index.month <= 3)]
            plot_segmented_line(
                ax,
                dates,
                [y - 0.25] * len(C),
                C * left_scaling,
                left_cmap,
                left_norm,
                lw=4,
            )
            # Resample to daily resolution.
            alt_C = right_vals[y].resample("D").mean()
            # Drop leap days from C
            alt_C = alt_C.loc[(alt_C.index.month != 2) | (alt_C.index.day != 29)]
            # Select only the period from October to March inclusive.
            alt_C = alt_C.loc[(alt_C.index.month >= 10) | (alt_C.index.month <= 3)]
            plot_segmented_line(
                ax,
                dates,
                [y + 0.25] * len(alt_C),
                alt_C * right_scaling,
                right_cmap,
                right_norm,
                lw=4,
            )
        # Draw a horizontal line indicating the duration of each period for the corresponding year, from "start" to "end" date.
        for i in periods.index:
            y = get_net_year(periods.loc[i, "start"])
            # Transpose period start and end to the 2014-2015 winter, then convert using mdates.date2num.
            start = periods.loc[i, "start"]
            start = mdates.date2num(
                dt.datetime(2014 + (start.year - y), start.month, start.day)
            )
            end = periods.loc[i, "end"]
            end = mdates.date2num(
                dt.datetime(2014 + (end.year - y), end.month, end.day)
            )
            ax.plot([start, end], [y - 0.12, y - 0.12], c="k", lw=3)
        # Do the same for the alternative periods
        if alt_periods is not None:
            for i in alt_periods.index:
                y = get_net_year(alt_periods.loc[i, "start"])
                # Transpose period start and end to the 2014-2015 winter, then convert using mdates.date2num.
                start = alt_periods.loc[i, "start"]
                start = mdates.date2num(
                    dt.datetime(2014 + (start.year - y), start.month, start.day)
                )
                end = alt_periods.loc[i, "end"]
                end = mdates.date2num(
                    dt.datetime(2014 + (end.year - y), end.month, end.day)
                )
                ax.plot([start, end], [y + 0.12, y + 0.12], c="royalblue", lw=3)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(year_selection[0] - 0.5, year_selection[-1] + 0.5)
        # Vertically flip y-axis
        ax.invert_yaxis()
        # Display all years as ticks.
        ax.set_yticks(year_selection)
        # Format each y tick as 2019/20 instead of 2019 (for example)
        ax.set_yticklabels([f"{str(y)[2:]}/{str(y + 1)[2:]}" for y in year_selection])
        # Format x ticks as month names.
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        # Remove tick marks
        ax.tick_params(axis="y", which="both", length=0)
        # Turn off axis frame
        ax.set_frame_on(False)
    # Add a colorbar.
    cax = fig.add_axes([0.15, 0, 0.3, 0.02])
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=left_cmap, norm=left_norm, orientation="horizontal"
    )
    cb.set_label(left_str)
    cb.set_ticks(left_ticks)
    cb.set_ticklabels([str(t) for t in left_ticks])
    cax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    cax.set_frame_on(False)
    # Add a colorbar.
    cax = fig.add_axes([0.57, 0, 0.3, 0.02])
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=right_cmap, norm=right_norm, orientation="horizontal"
    )
    cb.set_label(right_str)
    cb.set_ticks(right_ticks)
    cb.set_ticklabels([str(t) for t in right_ticks])
    cax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    cax.set_frame_on(False)
    axs[1].plot([], [], c="k", lw=2, label="System-defining events")
    if alt_periods is not None:
        axs[1].plot([], [], c="royalblue", lw=2, label="Alternative events")
    # Place legend to the right of the colour bar.
    axs[1].legend(loc="center left", bbox_to_anchor=(-0.5, -0.35), frameon=False)
    if save_fig:
        if path_str is None:
            print("Please provide a name for the saved figure.")
        else:
            fig.savefig(f"{path_str}.pdf", bbox_inches="tight")
    else:
        plt.show()
    return fig, axs


def plot_fc_util(
    n: pypsa.Network,
    cmap: mpl.colors.ListedColormap,
    norm: mpl.colors.BoundaryNorm,
    regions: gpd.GeoDataFrame,
    periods: pd.DataFrame,
    fc_flex: pd.DataFrame,
    ax: plt.Axes,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    cbar: bool = True,
    cbar_label: str = None,
    scaling_factor: float = 1,
):
    """Plot the fuel cell utilization for the given network and regions.

    Parameters:
    -----------
    n: pypsa.Network
        PyPSA network.
    cmap: mpl.colors.ListedColormap
        Colormap for the fuel cell utilization.
    norm: mpl.colors.BoundaryNorm
        Normalization for the colormap.
    regions: gpd.GeoDataFrame
        GeoDataFrame with the regions.
    periods: pd.DataFrame
        DataFrame with the system-defining events.
    fc_flex: pd.DataFrame
        DataFrame with the fuel cell flexibility.
    ax: plt.Axes
        Axes to plot on.
    projection: ccrs.Projection
        Projection for the plot.
    cbar: bool
        If True, add a colorbar to the plot.
    cbar_label: str
        Label for the colorbar.
    scaling_factor: float
        Scaling factor for the fuel cell utilization."""
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": projection},
            figsize=(6 * cm, 6 * cm),
        )
    n.plot(ax=ax, bus_sizes=0, line_widths=0, link_widths=0)

    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    df_fc = pd.DataFrame(index=r.index, columns=periods.index).astype(float)
    for i, period in periods.iterrows():
        start, end = period["start"], period["end"]
        df_fc[i] = fc_flex.loc[start:end].mean(axis=0)
    r["util"] = df_fc.mean(axis=1)
    r["util"] *= scaling_factor
    r.plot(
        ax=ax,
        column="util",
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    if cbar:
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.01,
            aspect=20,
            shrink=0.9,
            fraction=0.05,
        )
        cbar.set_label(f"{cbar_label}", fontsize=7)
        cbar.ax.tick_params(labelsize=7)


def plot_optimal_costs(
    opt_networks: dict,
    techs: list = [
        "variable",
        "H2",
        "battery",
        "transmission-ac",
        "transmission-dc",
        "onwind",
        "offwind",
        "solar",
    ],
    pretty_names: dict = {
        "variable": "Variable costs",
        "transmission-ac": "AC  transmission",
        "transmission-dc": "DC transmission",
        "onwind": "Onshore wind",
        "offwind": "Offshore wind",
        "solar": "Solar",
        "H2": "Hydrogen",
        "battery": "Battery",
    },
    storage_units: bool = False,
    save_fig: bool = False,
):
    """Plot the optimal costs for the different years.

    Parameters:
    -----------
    opt_networks: dict
        Dictionary of PyPSA networks.
    techs: list
        List of technologies.
    pretty_names: dict
        Pretty names for the technologies.
    storage_units: bool
        If True, use storage units instead of stores.
    save_fig: bool
        If True, save the figure.
    """
    opt_objs, _ = optimal_costs(opt_networks, techs, pretty_names, storage_units)
    n_cs = list(opt_networks.values())[0].carriers.color
    cs = [
        "#c0c0c0",
        n_cs["H2"],
        n_cs["battery"],
        "#70af1d",
        "#92d123",
        n_cs["onwind"],
        n_cs["offwind-ac"],
        n_cs["solar"],
    ]

    fig, ax = plt.subplots(1, 1, figsize=(32.0 * cm, 7 * cm))
    (opt_objs / 1e9).plot.bar(stacked=True, ax=ax, color=cs, width=0.7)

    # Labels
    ax.set_xlabel("Weather year")
    ax.set_ylabel("Annual system costs [billion EUR / a]")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reversed(handles),
        reversed(labels),
        bbox_to_anchor=(0, -0.25),
        loc="upper left",
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    # Ticks, grid
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.grid(color="lightgray", linestyle="solid", which="major")
    ax.yaxis.grid(color="lightgray", linestyle="dotted", which="minor")

    if save_fig:
        plt.savefig("../plots/optimal_costs.pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_period_anomalies(
    flex_anomaly: pd.DataFrame,
    system_anomaly: pd.DataFrame,
    periods: pd.DataFrame,
    tech: list = [
        "biomass",
        "nuclear",
        "H2 fuel cell",
        "battery discharger",
        "PHS",
        "hydro",
    ],
    tech_colours: list = [
        "#baa741",
        "#ff8c00",
        "#c251ae",
        "#ace37f",
        "#51dbcc",
        "#298c81",
    ],
    plot_all_system: bool = True,
    resampled: str = "1D",
    save_fig: bool = False,
    path_str: str = None,
):
    """Plot the flexibility and system anomalies for the different periods.

    Parameters:
    -----------
    flex_anomaly: pd.DataFrame
        Flexibility anomalies per technology.
    system_anomaly: pd.DataFrame
        System anomalies (net load, load, wind, solar).
    periods: pd.DataFrame
        System-defining events.
    tech: list
        List of technologies.
    tech_colours: list
        List of colours for the technologies.
    plot_all_system: bool
        If True, plot all system anomalies. Otherwise only net load.
    resampled: str
        Resampling frequency.
    save_fig: bool
        If True, save the figure.
    path_str: str
        Path to save the figure.
    """
    nb_plots = len(periods)
    nb_rows = nb_plots // 4 if nb_plots % 4 == 0 else nb_plots // 4 + 1
    fig, axs = plt.subplots(
        nb_rows,
        4,
        figsize=(30 * cm, nb_rows * 7 * cm),
        sharey=True,
        gridspec_kw={"hspace": 0.6},
    )
    for i, row in periods.iterrows():
        ax = axs.flatten()[i]
        start = row["start"]
        end = row["end"]
        # Plot stack plot of flexibility.
        p = (
            flex_anomaly.loc[start:end, tech].astype(float).resample(resampled).mean()
            / 1e3
        )  # in GW
        p_neg = p.clip(upper=0)
        p_pos = p.clip(lower=0)
        ax.stackplot(p_pos.index, p_pos.T, colors=tech_colours, labels=p_pos.columns)
        ax.stackplot(p_neg.index, p_neg.T, colors=tech_colours)
        # Plot net load anomaly.
        net_load = (
            system_anomaly.loc[start:end, "Net load anomaly"].resample(resampled).mean()
            / 1e3
        )
        ax.plot(
            net_load.index,
            net_load,
            color="black",
            lw=1,
            ls="--",
            label="Net load anomaly",
        )
        if plot_all_system:
            load = (
                system_anomaly.loc[start:end, "Load anomaly"].resample(resampled).mean()
                / 1e3
            )
            wind = (
                system_anomaly.loc[start:end, "Wind anomaly"].resample(resampled).mean()
                / 1e3
            )
            solar = (
                system_anomaly.loc[start:end, "Solar anomaly"]
                .resample(resampled)
                .mean()
                / 1e3
            )
            ax.plot(load.index, load, color="red", lw=1, ls=":", label="Load anomaly")
            ax.plot(wind.index, wind, color="blue", lw=1, ls=":", label="Wind anomaly")
            ax.plot(
                solar.index, solar, color="grey", lw=1, ls=":", label="Solar anomaly"
            )
        ax.set_title(f"Event {i}")
        ax.set_ylabel("Flexibility/Anomaly [GW]")
        ax.set_xlabel(f"{start.date()} - {end.date()}", fontsize=8)
        # Set x-ticks to only show day and month
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        # Change font size of x tick labels.
        ax.tick_params(axis="x", labelsize=8, rotation=90)
    # Add legend to the last row between the 2nd and 3rd plot.
    labels, handles = ax.get_legend_handles_labels()
    axs.flatten()[-3].legend(
        labels, handles, loc="center", bbox_to_anchor=(1, -0.6), frameon=False, ncols=4
    )
    # Hide empty plots.
    for ax in axs.flatten()[nb_plots:]:
        ax.axis("off")
    if save_fig:
        if path_str is None:
            print("Please specify a path to save the figure.")
        else:
            plt.savefig(f"{path_str}.pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_prices(
    n: pypsa.Network,
    cmap: mpl.colors.ListedColormap,
    norm: mpl.colors.BoundaryNorm,
    regions: gpd.GeoDataFrame,
    periods: pd.DataFrame,
    nodal_prices: pd.DataFrame,
    ax: plt.Axes,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    cbar: bool = True,
    cbar_label: str = None,
):
    """Plot the nodal prices of the network on a map, with regions coloured according to their average prices during system-defining events.

    Parameters:
    -----------
    n: pypsa.Network
        The PyPSA network.
    cmap: mpl.colors.ListedColormap
        Colormap for the regions.
    norm: mpl.colors.BoundaryNorm
        Normalization for the colormap.
    regions: gpd.GeoDataFrame
        GeoDataFrame containing the regions to plot.
    periods: pd.DataFrame
        DataFrame containing the system-defining events.
    nodal_prices: pd.DataFrame
        DataFrame containing the nodal prices for each period.
    ax: plt.Axes
        Axes to plot on.
    projection: ccrs.Projection
        Projection to use for the map.
    cbar: bool
        Whether to add a colorbar.
    cbar_label: str
        Label for the colorbar.
    """
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": projection},
            figsize=(6 * cm, 6 * cm),
        )
    n.plot(ax=ax, bus_sizes=0, line_widths=0, link_widths=0)

    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)
    event_prices = pd.DataFrame(index=r.index, columns=periods.index).astype(float)
    for i, period in periods.iterrows():
        event_prices[i] = nodal_prices.loc[period["start"] : period["end"]].mean(axis=0)
    r["prices"] = event_prices.mean(axis=1)
    r.plot(
        ax=ax,
        column="prices",
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    if cbar:
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.01,
            aspect=20,
            shrink=0.9,
            fraction=0.05,
        )
        cbar.set_label(f"{cbar_label}", fontsize=7)
        cbar.ax.tick_params(labelsize=7)


def plot_scatter_quadrants(
    marked_years, opt_objs, share_unserved, max_unserved, title, colour="#2F3C30"
):
    """Plot scatter plots of total costs against (maximal) unserved energy, once for design years and operational years each in a 2x2 grid.

    Parameters:
    -----------
    marked_years: list
        List of years to be highlighted in the scatter plot.
    opt_objs: dict
        Total system costs
    share_unserved: pd.DataFrame
        Share of unserved energy for each year.
    max_unserved: pd.DataFrame
        Maximum unserved load for each year.
    title: str
        Title of the figure.
    colour: str
        Colour to use for the highlighted years in the scatter plot.
    """
    fig, axs = plt.subplots(
        2, 2, figsize=(18 * cm, 14 * cm), gridspec_kw={"hspace": 0.3}
    )
    fig.suptitle(title, fontsize=8)
    total_costs = opt_objs["total"] / 1e9  # in bn EUR

    # Unserved energy (in percent)
    share_unserved_reorder = share_unserved.mean(axis="columns")
    op_share_unserved_reorder = share_unserved.mean(axis="index")
    share_unserved_reorder.index = total_costs.index
    op_share_unserved_reorder.index = total_costs.index

    # Max unserved load (in GW)
    max_unserved_reorder = max_unserved.mean(axis="columns")
    op_max_unserved_reorder = max_unserved.mean(axis="index")
    max_unserved_reorder.index = total_costs.index
    op_max_unserved_reorder.index = total_costs.index

    ax = axs[0, 0]
    # Scatter plot.
    ax.scatter(total_costs, share_unserved_reorder, color="#2F3C30", alpha=0.2, s=20)

    # Add correlation.
    corr = total_costs.corr(share_unserved_reorder)
    ax.text(
        0.6,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    for i, txt in enumerate(total_costs.index):
        if txt in marked_years:
            ax.annotate(
                txt,
                (total_costs[i] - 0.5, share_unserved_reorder[i]),
                fontsize=7,
                ha="right",
                va="bottom",
                color="black",
            )
    # Scatter plot only those above years with full opacity
    ax.scatter(
        total_costs[marked_years],
        share_unserved_reorder[marked_years],
        color=colour,
        alpha=0.9,
        s=20,
        zorder=3,
    )

    # ax.hlines(0.05, color="red", linestyle=":", linewidth=0.5, label="ENTSO-E reliability")

    ax.set_ylabel("Share of expected unserved energy [%]", fontsize=8)
    ax.set_title("Design year", fontsize=8)
    ax = axs[1, 0]
    # Scatter plot.
    ax.scatter(total_costs, op_share_unserved_reorder, color="#2F3C30", alpha=0.2, s=20)

    # Add correlation.
    corr = total_costs.corr(op_share_unserved_reorder)
    ax.text(
        0,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    # Annotate some years such as "62/63", "41/42", "19/20", "14/15", and "65/66"
    for i, txt in enumerate(total_costs.index):
        if txt in marked_years:
            ax.annotate(
                txt,
                (total_costs[i] - 0.5, op_share_unserved_reorder[i]),
                fontsize=7,
                ha="right",
                va="bottom",
                color="black",
            )
    # Scatter plot only those above years with full opacity
    ax.scatter(
        total_costs[marked_years],
        op_share_unserved_reorder[marked_years],
        color=colour,
        alpha=0.9,
        s=20,
        zorder=3,
    )

    ax.set_ylabel("Share of expected unserved energy [%]", fontsize=8)
    ax.set_title("Stress test: Operational year", fontsize=8)
    ax = axs[0, 1]
    # Scatter plot.
    ax.scatter(total_costs, max_unserved_reorder, color="#2F3C30", alpha=0.2, s=20)
    # Add correlation.
    corr = total_costs.corr(max_unserved_reorder)
    ax.text(
        0.6,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    for i, txt in enumerate(total_costs.index):
        if txt in marked_years:
            ax.annotate(
                txt,
                (total_costs[i] - 0.5, max_unserved_reorder[i]),
                fontsize=7,
                ha="right",
                va="bottom",
                color="black",
            )
    ax.scatter(
        total_costs[marked_years],
        max_unserved_reorder[marked_years],
        color=colour,
        alpha=0.9,
        s=20,
        zorder=3,
    )
    ax.set_ylabel("Expected max. unserved load [GW]", fontsize=8)
    ax.set_title("Design year", fontsize=8)
    ax = axs[1, 1]
    # Scatter plot.
    ax.scatter(total_costs, op_max_unserved_reorder, color="#2F3C30", alpha=0.2, s=20)
    # Add correlation.
    corr = total_costs.corr(op_max_unserved_reorder)
    ax.text(
        0,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    for i, txt in enumerate(total_costs.index):
        if txt in marked_years:
            ax.annotate(
                txt,
                (total_costs[i] - 0.5, op_max_unserved_reorder[i]),
                fontsize=7,
                ha="right",
                va="bottom",
                color="black",
            )
    # Scatter plot only those above years with full opacity
    ax.scatter(
        total_costs[marked_years],
        op_max_unserved_reorder[marked_years],
        color=colour,
        alpha=0.9,
        s=20,
        zorder=3,
    )
    ax.set_ylabel("Expected max. unserved load [GW]", fontsize=8)
    ax.set_title("Stress test: Operational year", fontsize=8)
    for ax in axs[:, 0]:
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    for ax in axs[:, 1]:
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    for ax in axs.flatten():
        # Labels
        ax.set_xlabel("Annual system costs [bn EUR / a]", fontsize=8)
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        sns.despine(ax=ax, left=True, bottom=True)
        ax.tick_params(axis="y", which="both", length=0, labelsize=7, rotation=0)
        ax.tick_params(axis="x", which="both", length=0, labelsize=7, rotation=0)
        # Ticks, grid

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.grid(color="lightgray", linestyle="solid", which="major")
        ax.xaxis.grid(color="lightgray", linestyle="dotted", which="minor")
        ax.legend(
            handles=handles,
            labels=labels,
            ncol=4,
            loc="upper left",
            fontsize=7,
            bbox_to_anchor=(0.1, -0.25),
            borderaxespad=0.0,
            frameon=False,
            labelspacing=0.75,
        )
        ax.set_xlim(120, 220)
        ax.yaxis.grid(color="lightgray", linestyle="solid", which="major")
        ax.yaxis.grid(color="lightgray", linestyle="dotted", which="minor")
    return fig, axs


def plot_scatter(ax, x_data, y_data, x_label, y_label, title, color="blue"):
    """Plot a scatter plot with correlation coefficient annotated."""
    ax.scatter(x_data, y_data, s=1, color=color)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)
    corr = x_data.corr(y_data)
    ax.annotate(
        f"Corr: {corr:.2f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
    )


def plot_segmented_line(ax, x, y, c, cmap, norm, **kwargs):
    """Plot a segmented horizontal line (using LineCollection) where each segment is coloured according to total_costs[y]"""
    segments = np.array([x[:-1], y[:-1], x[1:], y[1:]]).T.reshape(-1, 2, 2)
    # Log norm
    lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
    lc.set_array(c)
    ax.add_collection(lc)
    return lc


def plot_total_costs(
    n: pypsa.Network,
    cmap: mpl.colors.ListedColormap,
    norm: mpl.colors.BoundaryNorm,
    regions: gpd.GeoDataFrame,
    periods: pd.DataFrame,
    nodal_costs: pd.DataFrame,
    ax: plt.Axes,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    cbar: bool = True,
    cbar_label: str = None,
):
    """Plot the total costs of the network on a map, with regions coloured according to their average costs during system-defining events.

    Parameters:
    -----------
    n: pypsa.Network
        The PyPSA network.
    cmap: mpl.colors.ListedColormap
        Colormap for the regions.
    norm: mpl.colors.BoundaryNorm
        Normalization for the colormap.
    regions: gpd.GeoDataFrame
        GeoDataFrame containing the regions to plot.
    periods: pd.DataFrame
        DataFrame containing the system-defining events.
    nodal_costs: pd.DataFrame
        DataFrame containing the nodal costs for each period.
    ax: plt.Axes
        Axes to plot on.
    projection: ccrs.Projection
        Projection to use for the map.
    cbar: bool
        Whether to add a colorbar.
    cbar_label: str
        Label for the colorbar.
    """
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": projection},
            figsize=(6 * cm, 6 * cm),
        )
    n.plot(ax=ax, bus_sizes=0, line_widths=0, link_widths=0)

    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    event_costs = pd.DataFrame(index=r.index, columns=periods.index).astype(float)
    for i, period in periods.iterrows():
        event_costs[i] = nodal_costs.loc[period["start"] : period["end"]].mean(axis=0)
    r["costs"] = event_costs.mean(axis=1)

    r.plot(
        ax=ax,
        column="costs",
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    if cbar:
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.01,
            aspect=20,
            shrink=0.9,
            fraction=0.05,
        )
        cbar.set_label(f"{cbar_label}", fontsize=7)
        cbar.ax.tick_params(labelsize=7)


def ranges_across_years(
    df: pd.DataFrame, years: list, resample: str = None, clip: bool = False
):
    """Create a DataFrame with the ranges of the given DataFrame across the specified years.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame with the data to be processed.
    years: list
        List of years to consider for the ranges.
    resample: str, optional
        Resampling frequency, e.g. '1D' for daily averages.
    clip: bool, optional
        If True, clip the values to be non-negative.
    """
    df_help = df.copy()
    df_help.index = range(len(df_help))
    list_help = [
        df_help.loc[i * 8760 : (i + 1) * 8760 - 1] for i in range(len(df_help) // 8760)
    ]
    for li in list_help:
        li.index = range(8760)
    full_df = pd.concat(list_help, axis=1)
    full_df.index = pd.date_range(
        f"{years[0]}-07-01", f"{years[0] + 1}-06-30 23:00", freq="h"
    )
    if clip:
        full_df = full_df.clip(lower=0)
    if resample is not None:
        full_df = full_df.resample(resample).mean()
    full_df.columns = years
    return full_df


#### CURRENTLY NOT IN USE:
def compute_incidence_matrix(periods, sensitivity_periods):
    """Compute the incidence matrix for the periods and sensitivity periods.

    Parameters:
    -----------
    periods: pd.DataFrame
        DataFrame with the periods.
    sensitivity_periods: dict
        Dictionary with the sensitivity periods, where keys are scenario names and values are DataFrames of periods.

    Returns:
    --------
    np.ndarray
        Incidence matrix where rows correspond to periods and columns to sensitivity periods.
    """
    matrix = np.zeros((len(periods), len(sensitivity_periods)))
    for j, scenar in enumerate(sensitivity_periods.keys()):
        alt_periods = sensitivity_periods[scenar]
        if len(alt_periods) == 0:
            matrix[:, j] = 0
        else:
            for i, period in periods.iterrows():
                start = period.start.tz_localize(tz="UTC")
                end = period.end.tz_localize(tz="UTC")
                time_slice = pd.date_range(start, end, freq="h")
                for alt_period in alt_periods.iterrows():
                    alt_time_slice = pd.date_range(
                        alt_period[1].start, alt_period[1].end, freq="h"
                    )
                    if len(set(time_slice).intersection(set(alt_time_slice))) > 0:
                        matrix[i, j] = 1
                        break
    return matrix


def plot_flex_events(
    periods: pd.DataFrame,
    all_flex: pd.DataFrame,
    avg_flex: pd.DataFrame,
    rolling: int = 24,
    mark_sde: bool = True,
    window_length: pd.Timedelta = pd.Timedelta("30d"),
    tech: list = [
        "DC",
        "AC",
        "biomass",
        "nuclear",
        "ror",
        "fuel_cells",
        "battery",
        "phs",
        "hydro",
    ],
    title: str = "Flexibility usage",
    save_fig: bool = False,
    path_str: str = None,
):
    """Plot flexibility usage of different technologies around the system-defining events.

    Parameters:
    -----------
    periods: pd.DataFrame
        DataFrame containing the start and end dates of the events.
    all_flex: pd.DataFrame
        DataFrame containing the flexibility usage of different technologies.
    avg_flex: pd.DataFrame
        DataFrame containing the average flexibility usage of different technologies.
    rolling: int
        Rolling window for the flexibility usage.
    mark_sde: bool
        Whether to mark the system-defining events.
    window_length: pd.Timedelta
        Length of the window around the events.
    tech: list
        List of technologies to plot.
    title: str
        Title of the plot.
    save_fig: bool
        Whether to save the figure.
    path_str: str
        Path to save the figure in.
    """
    colours = {
        "DC": "#8a1caf",
        "AC": "#70af1d",
        "biomass": "#baa741",
        "nuclear": "#ff8c00",
        "ror": "#3dbfb0",
        "fuel_cells": "#c251ae",
        "battery": "#ace37f",
        "phs": "#51dbcc",
        "hydro": "#298c81",
    }

    nrows = len(periods) // 4 if len(periods) % 4 == 0 else len(periods) // 4 + 1
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=4,
        figsize=(30 * cm, 50 * cm),
        sharey=True,
        gridspec_kw={"hspace": 0.5},
    )
    fig.suptitle(title, fontsize=12)
    # No vertical space between title and the rest of the plot.
    fig.subplots_adjust(top=0.95)
    for i, row in periods.iterrows():
        window_start = row["start"] - window_length
        window_end = row["end"] + window_length
        ax = axs.flatten()[i]
        ax.plot(
            all_flex.loc[window_start:window_end, tech].rolling(rolling).mean().index,
            all_flex.loc[window_start:window_end, tech].rolling(rolling).mean(),
            color=[colours[t] for t in tech] if len(tech) > 1 else colours[tech[0]],
            lw=0.5,
        )
        # Translated 2014-2015 from avg_flex to correct year.
        shifted_avg_flex = avg_flex.copy()
        year = get_net_year(row["start"])
        new_index = pd.date_range(
            f"{year}-07-01", f"{year + 1}-06-30 23:00:00", freq="h"
        )
        if pd.Timestamp(f"{year + 1}-01-01").is_leap_year:
            new_index = new_index.drop(
                pd.date_range(
                    f"{year + 1}-02-29", f"{year + 1}-02-29 23:00:00", freq="h"
                )
            )
        shifted_avg_flex.index = new_index
        ax.plot(
            shifted_avg_flex.loc[window_start:window_end, tech]
            .rolling(rolling)
            .mean()
            .index,
            shifted_avg_flex.loc[window_start:window_end, tech].rolling(rolling).mean(),
            color=[colours[t] for t in tech] if len(tech) > 1 else colours[tech[0]],
            lw=0.25,
            ls="dashed",
        )
        if mark_sde:
            ax.fill_between(
                all_flex.loc[row["start"] : row["end"], tech]
                .rolling(rolling)
                .mean()
                .index,
                all_flex.loc[window_start:window_end, tech]
                .rolling(rolling)
                .mean()
                .min(),
                all_flex.loc[window_start:window_end, tech]
                .rolling(rolling)
                .mean()
                .max(),
                color="gray",
                alpha=0.2,
            )
        ax.set_title(f"Event {i}")
        # ax.set_ylim(0, 1)
        ax.set_xlim(window_start, window_end)
        # Only mark beginning of months in x-tickmarkers.
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
        # ax.set_xlabel("Time")
        ax.set_ylabel("Flexibility usage")
    if save_fig:
        if path_str is None:
            print("Please specify a path to save the figure.")
        else:
            plt.savefig(f"{path_str}.pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_incidence_matrix(
    m, sensitivity_periods, cmap=mpl.colors.ListedColormap(["red", "green"]), ax=None
):
    """Plot the incidence matrix of system-defining events and sensitivity scenarios.

    Parameters:
    -----------
    m: pd.DataFrame
        Incidence matrix with system-defining events as rows and sensitivity scenarios as columns.
    sensitivity_periods: dict
        Dictionary with sensitivity scenarios, where keys are tuples of (year, month, day) and values are the scenario names.
    cmap: mpl.colors.ListedColormap
        Colormap for the matrix.
    ax: plt.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    """
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(m, cmap=cmap)
    ax.set_xlabel("Sensitivity scenarios")
    ax.set_ylabel("System-defining events")
    ax.set_xticks(np.arange(len(sensitivity_periods)))
    ax.set_xticklabels(
        [
            f"{scenar[0]}_{scenar[1]}_{scenar[2]}"
            for scenar in sensitivity_periods.keys()
        ],
        rotation=90,
    )
    return im


def plot_generation_stack(
    n: pypsa.Network,
    start: pd.Timestamp,
    end: pd.Timestamp,
    periods: pd.DataFrame,
    freq: str = "1D",
    new_index: pd.TimedeltaIndex = None,
    ax=None,
):
    """Plot the generation stack with highlighted difficult periods.

    Parameters:
    -----------
    n: PyPSA network.
    start: pd.Timestamp
        Start date.
    end: pd.Timestamp
        End date.
    periods: pd.DataFrame
        System-defining events.
    freq: str
        Resampling frequency.
    new_index: pd.TimedeltaIndex
        New index for the data.
    ax: matplotlib.axes
        Axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
    # Gather generation, storage, and load data.
    # NB: Transmission cancels out on the European level.
    p_gen = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum() / 1e3
    p_store = n.stores_t.p.groupby(n.stores.carrier, axis=1).sum() / 1e3
    p_storage = n.storage_units_t.p.groupby(n.storage_units.carrier, axis=1).sum() / 1e3
    p = pd.concat([p_gen, p_store, p_storage], axis=1)
    p = p.resample(freq).mean()
    # Ensure we have no leap days.
    p = p[~((p.index.month == 2) & (p.index.day == 29))]
    p_neg = p.clip(upper=0)
    p = p.clip(lower=0)
    loads = n.loads_t.p_set.sum(axis=1).resample(freq).mean() / 1e3
    # Ensure we have no leap days.
    loads = loads[~((loads.index.month == 2) & (loads.index.day == 29))]
    # Ensure load shedding has a colour.
    n.carriers.color["load-shedding"] = "#000000"
    colors = [n.carriers.color[carrier] for carrier in p.columns]
    if new_index is not None:
        # Reindex the dates, as the years are wrong.
        p.index = new_index
        p_neg.index = new_index
        loads.index = new_index
    # If we specified a start and end date, only plot that period.
    p_slice = p.loc[start:end]
    p_neg_slice = p_neg.loc[start:end]
    loads = loads[start:end]
    # Plot the generation stack.
    ax.stackplot(p_slice.index, p_slice.transpose(), colors=colors, labels=p.columns)
    ax.stackplot(p_neg_slice.index, p_neg_slice.transpose(), colors=colors)
    # Plot the SDEs.
    # First the ones we have identified.
    ymin, ymax = ax.get_ylim()
    for _, row in periods.iterrows():
        if (
            row["start"].tz_localize(None) > start
            and row["end"].tz_localize(None) < end
        ):
            period_start = pd.Timestamp(row["start"].date())
            period_end = pd.Timestamp(row["end"].date())
            ax.fill_between(
                pd.DatetimeIndex(pd.date_range(period_start, period_end, freq="D")),
                ymin,
                ymax,
                color="grey",
                alpha=0.3,
                label="Filtered period",
            )
    # Load.
    ax.plot(loads, ls="dashed", color="red", label="load", linewidth=0.5)
    ax.set_xlim(start, end)
    if ax is None:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("GW")
    plt.tight_layout()


def load_sensitivity_periods(config: dict):
    """Load the system-defining events based on the configuration."""
    scen_name = config["difficult_periods"]["scen_name"]
    clusters = config["scenario"]["clusters"]
    ll = config["scenario"]["ll"]
    opts = config["scenario"]["opts"]
    periods_name = f"sde_{scen_name}_{clusters[0]}_elec_l{ll[0]}_{opts[0]}"

    periods = pd.read_csv(
        f"../pypsa-eur/results/periods/{periods_name}.csv",
        index_col=0,
        parse_dates=["start", "end", "peak_hour"],
    )

    for col in ["start", "end", "peak_hour"]:
        periods[col] = periods[col].dt.tz_localize(None)
    return periods
