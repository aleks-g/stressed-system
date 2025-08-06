# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Generate data for the notebooks in which we analyse and plot the unserved energy.
"""

import pandas as pd
from _notebook_utilities import *
import logging
# Suppress warnings and info messages from 'pypsa.io'
logging.getLogger("pypsa.io").setLevel(logging.ERROR)


if __name__ == "__main__":
    overwrite_data = True

    # Load config file of the scenario with no transmission expansion, as otherwise unserved energy is degenerately high due to small shifts in transmission expansion in the default scenario.
    # Need to deactivate/comment out all scenarios except ll=c1.0, and opts=Co2L0.0 in config/stressful-weather-sensitivities.yaml.
    config_name = "stressful-weather-sensitivities"
    config, scenario_def, years, opt_networks = load_opt_networks(config_name, config_str = "base_s_90_elec_lc1.0_Co2L0.0", load_networks=True)

    # Load default configuration to obtain all other data.
    original_config, original_scenario_def, original_years, original_opt_networks = load_opt_networks("stressful-weather", config_str = "base_s_90_elec_lc1.25_Co2L0.0", load_networks=False)

    # Use processing data from original/default config (need to run _generate_data_for_analysis.py to generate these).
    folder = "processing_data/stressful-weather"

    periods = load_periods(original_config)
    opt_objs = pd.read_csv(f"{folder}/opt_objs.csv", index_col=0)
    reindex_opt_objs = opt_objs.copy().sum(axis="columns") 
    reindex_opt_objs.index=years

    # Winter load, winter wind cf
    net_load = pd.read_csv(f"{folder}/net_load.csv", index_col=0, parse_dates=True)
    total_load = pd.read_csv(f"{folder}/total_load.csv", index_col=0, parse_dates=True)
    winter_load = pd.read_csv(f"{folder}/winter_load.csv", index_col=0)
    annual_cfs = pd.read_csv(f"{folder}/annual_cfs.csv", index_col=0)

    # Get clustering data.
    clusters = pd.read_csv("clustering/stressful-weather/clustered_vals_4.csv", index_col=0)

    share_unserved = pd.DataFrame(index=years, columns=years).astype(float)
    max_unserved = pd.DataFrame(index=years, columns=years).astype(float)
    unserved = pd.DataFrame(index=years, columns=years).astype(float)

    for year in years:
        for op_year in years:
            if year != op_year:
                loadshedding = pd.read_csv(f"../pypsa-eur/results/{config_name}/weather_year_{year}/validation/weather_year_{op_year}_base_s_90_elec_lc1.0_Co2L0.0.csv", index_col=0, parse_dates=True)
                annual_load = total_load.loc[f"{op_year}-07-01":f"{op_year+1}-06-30", "0"].sum() / 1e3 # in GWh
                unserved.loc[year, op_year] = loadshedding.sum().sum() / 1e6  # in GWh
                share_unserved.loc[year, op_year] = 100 *(loadshedding/1e6).sum().sum() / annual_load
                max_unserved.loc[year, op_year] = (loadshedding/1e6).sum(axis="columns").max()
    if overwrite_data:
        share_unserved.to_csv(f"processing_data/{config_name}/share_unserved.csv")
        max_unserved.to_csv(f"processing_data/{config_name}/max_unserved.csv")
        unserved.to_csv(f"processing_data/{config_name}/unserved.csv")


    # Rank years by difficulty of design year and operational year and compare whether these match.
    ranked_years = pd.DataFrame(index = years, columns = ["System costs", "Lowest annual solar CF", "Lowest annual wind CF", "Winter load", "Causes deficit", "Prevents deficits", "Highest net load", "Causes peaks", "Prevents peaks", "SDE"]).astype(float)

    ranked_years["System costs"] = reindex_opt_objs.sort_values().rank(ascending=False).astype(int)
    ranked_years["Lowest annual solar CF"] = annual_cfs.sort_values("solar").rank(ascending=True)["solar"].astype(int)
    ranked_years["Lowest annual wind CF"] = annual_cfs.sort_values("wind").rank(ascending=True)["wind"].astype(int)
    ranked_years["Winter load"] = winter_load.sort_values("load").rank(ascending=False).astype(int)

    # For deficits, need to look at share of unserved energy.
    design_sorted_unserved = share_unserved.T.describe().mean().sort_values().rank()
    op_sorted_unserved = share_unserved.describe().mean().sort_values().rank(ascending=False)

    # For peaks, need to look at maximal unserved energy.
    design_peak_unserved = max_unserved.T.describe().mean().sort_values().rank()
    op_peak_unserved = max_unserved.describe().mean().sort_values().rank(ascending=False)

    # Reindex dataframes
    for df in [design_sorted_unserved, op_sorted_unserved, design_peak_unserved, op_peak_unserved]:
        df.index = [int(i) for i in df.index]

    ranked_years["Causes deficit"] = design_sorted_unserved.astype(int)
    ranked_years["Prevents deficits"] = op_sorted_unserved.astype(int)

    for year in years:
        ranked_years.loc[year, "Highest net load"] = net_load.loc[f"{year}-07-01":f"{year+1}-06-30 23:00", "Net load"].max()
    ranked_years["Highest net load"] = ranked_years["Highest net load"].sort_values().rank(ascending=False).astype(int)
    ranked_years["Causes peaks"] = design_peak_unserved.astype(int)
    ranked_years["Prevents peaks"] = op_peak_unserved.astype(int)

    ranked_years["SDE"] = 0
    for i, period in periods.iterrows():
        net_year = get_net_year(period.start)
        ranked_years.loc[net_year, "SDE"] = 1

    if overwrite_data:
        ranked_years.to_csv(f"processing_data/{config_name}/ranked_years.csv")
