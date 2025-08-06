# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Cluster SDEs according to indicators based on system data.
"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from _notebook_utilities import *
import logging
# Suppress warnings and info messages from 'pypsa.io'
logging.getLogger("pypsa.io").setLevel(logging.ERROR)


if __name__ == "__main__":
    # Settings for clustering
    n_clusters = 4 # Determined by Silhouette score and Calinski-Harabasz score.
    save_fig = True

    # Initiation, load data
    config_name = "stressful-weather"
    config, scenario_def, years, opt_networks = load_opt_networks(config_name, load_networks=False)
    periods = load_periods(config)
    stats_periods = pd.read_csv(f"processing_data/{config_name}/stats_periods.csv", index_col=0, parse_dates=True)

    # Compute duration of each period
    for i, period in periods.iterrows():
        stats_periods.loc[i, "duration"] = (pd.Timestamp(period.end) - pd.Timestamp(period.start)).total_seconds() / 3600

    # Drop unnecessary columns and reorder.
    heatmap = stats_periods.drop(columns=["start", "end", "peak_hour", "net_load_peak_hour", "wind_cf", "energy_deficit"])
    heatmap = heatmap[["highest_net_load", "avg_net_load", "duration", "h2_discharge", "max_fc_discharge", "avg_rel_load", "wind_anom", "annual_cost"]]
    # Do not use annual costs for clustering SDEs, as this is independent of the event.
    clustered_vals = heatmap.copy().drop(columns=["annual_cost"])
    # Normalize the values
    normalized_vals = (clustered_vals - clustered_vals.mean()) / clustered_vals.std()

    # Use elbow method to determine optimal number of clusters, but also look at variance-ratio criterion of Calinski and Harabisz (https://arxiv.org/pdf/2212.12189)
    inertia = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(normalized_vals)
        cluster_labels = kmeans.fit_predict(normalized_vals)
        silhouette_avg = silhouette_score(normalized_vals, cluster_labels)
        print(f"Number of clusters: {n}, silhouette score: {silhouette_avg}")
        print(f"Calinski-Harabasz score: {calinski_harabasz_score(normalized_vals, cluster_labels)}")
        # Check Bayesian Information Criterion
        inertia.append(kmeans.inertia_)
    # Can plot BIC to find elbow
    fig, ax = plt.subplots(1, 1, figsize=(10 * cm, 10 * cm))
    ax.plot(range(2, 11), inertia, marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    fig.savefig(f"clustering/{config_name}/elbow_method.png", bbox_inches="tight", dpi=300) if save_fig else None

    # Rank all events according to category
    ranked_heatmap = heatmap.rank(ascending=False)
    # Replace wind_anom with ascending=True values
    ranked_heatmap["wind_anom"] = heatmap["wind_anom"].rank(ascending=True)

    ## CLUSTERING
    # Fit the KMeans model
    kmeans_vals = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_vals)
    # With the fixed random state and n_clusters = 4, the corresponding clusters are: {0: "C", 1: "E", 2: "S", 3: "P"}

    # Add the cluster labels to the DataFrame
    clustered_vals['cluster'] = kmeans_vals.labels_

    # Print the cluster centroids
    cluster_centroids = pd.DataFrame(kmeans_vals.cluster_centers_ * clustered_vals.std()[:-1].values + clustered_vals.mean()[:-1].values, columns=clustered_vals.columns[:-1])


    # Save results
    folder = f"clustering/{config_name}"
    # RANKED HEATMAP / KPI
    ranked_heatmap.to_csv(f"{folder}/kpi.csv")
    # CLUSTERS
    clustered_vals.to_csv(f"{folder}/clustered_vals_{n_clusters}.csv")
    # PRINT CENTROIDS
    cluster_centroids.to_csv(f"{folder}/centroids_{n_clusters}.csv")



