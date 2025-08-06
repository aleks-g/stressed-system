# SPDX-FileCopyrightText: 2025 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Functions for plotting affected areas on maps based on clustered data."""

import pypsa

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, Patch

import pandas as pd
import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
import seaborn as sns

from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import geopy.distance

from _notebook_utilities import *

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="facecolor will have no effect as it has been defined as",
)

def in_or_on_hull(p: np.ndarray, hull: ConvexHull, threshold: float = 0) -> bool:
    """Check if a point is in or on the convex hull.
    
    Parameters:
    -----------
    p: np.ndarray
        The point to check
    hull: ConvexHull
        The convex hull
    threshold: float
        Threshold for considering a point on the hull
        
    Returns:
    --------
    bool
        True if the point is in or on the hull, False otherwise
    """
    from matplotlib.path import Path
    hull_path = Path(hull.points[hull.vertices])
    return hull_path.contains_point(p, radius=threshold)

def draw_region(
    shape: gpd.GeoDataFrame,
    cluster_nr: int,
    T: float,
    print_progress: bool = False,
) -> tuple[gpd.GeoDataFrame, ConvexHull]:
    """Draw a region based on the clustered data.
    
    Parameters:
    -----------
    shape: gpd.GeoDataFrame
        GeoDataFrame with clustered data
    cluster_nr: int
        Cluster number to draw
    T: float
        Threshold for the ratio of nodes in the cluster to all nodes in the region
    print_progress: bool
        Whether to print progress information
    
    Returns:
    --------
    tuple[gpd.GeoDataFrame, ConvexHull]
        The subregions and the convex hull
    """
    # Define first convex hull to include all points in the selected cluster
    subnodes = shape[shape.cluster == cluster_nr]
    centroid = subnodes[["x", "y"]].mean()
    hull = ConvexHull(subnodes[["x", "y"]])

    # Find all subregions which lie within the convex hull.
    subregions = []
    for i in shape.index:
        if in_or_on_hull(shape.loc[i][["x", "y"]].values, hull):
            subregions.append(i)
    subregions = shape.loc[subregions]

    # Compute first threshold.
    threshold = len(subregions[subregions.cluster == cluster_nr])/len(subregions)
    iters = 0

    # Iteratively reduce the convex hull to meet threshold.
    while threshold < T and iters < 20:
        if print_progress:
            print(f"Reduce cluster, as threshold {T} is not met; current ratio: {threshold} for {len(subregions)} included nodes.")
        helper_df = subregions[subregions.cluster == cluster_nr].copy()
        # Sort all nodes inside the chosen cluster by distance from the centroid.
        helper_df["distance"] = helper_df.apply(lambda x: geopy.distance.distance((x.y, x.x), (centroid.y, centroid.x)).m, axis=1)
        helper_df = helper_df.sort_values("distance", ascending=False)
        # Remove the node furthest away from the centroid.
        helper_df.drop(helper_df.index[0], inplace=True)
        # Recompute the centroid.
        centroid = helper_df[["x", "y"]].mean()
        # Recompute the convex hull.
        hull = ConvexHull(helper_df[["x", "y"]])

        # Find all subregions which lie within the convex hull.
        subregions = []
        for i in shape.index:
            if in_or_on_hull(shape.loc[i][["x", "y"]].values, hull):
                subregions.append(i)
        subregions = shape.loc[subregions]
        threshold = len(subregions[subregions.cluster == cluster_nr])/len(subregions)
        iters += 1
        if iters == 20:
            if print_progress:
                print("Max iterations reached.")
            break
    if print_progress:
        print(f"Threshold: {threshold} with {len(subregions)} included clusters.")
    
    # Check if it's possible to add any neighbouring nodes within our cluster without violating the threshold.
    helper_df = subregions.copy()
    outside_cluster = shape[shape.cluster == cluster_nr].index.difference(subregions.index)
    for i in outside_cluster:
        helper_df.loc[i] = shape.loc[i]
        temp_hull = ConvexHull(helper_df[["x", "y"]])
        subr_tmp = []
        for j in shape.index:
            if in_or_on_hull(shape.loc[j][["x", "y"]].values, temp_hull):
                subr_tmp.append(j)
        subr_tmp = shape.loc[subr_tmp]
        if len(subr_tmp[subr_tmp.cluster == cluster_nr])/len(subr_tmp) > threshold:
            subregions = subr_tmp
            threshold = len(subregions[subregions.cluster == cluster_nr])/len(subregions)
            hull = temp_hull
    if print_progress:
        print(f"Final threshold: {threshold} with {len(subregions)} included clusters.")
    return subregions, hull

def select_clusters(
    r: gpd.GeoDataFrame,
    column: str,
    cluster_nb: int,
    cluster_sense: str,
) -> tuple[gpd.GeoDataFrame, int]:
    """Select clusters based on the data.
    
    Parameters:
    -----------
    r: gpd.GeoDataFrame
        GeoDataFrame with data
    column: str
        Column to cluster on
    cluster_nb: int
        Number of clusters
    cluster_sense: str
        Whether to select the max or min cluster
        
    Returns:
    --------
    tuple[gpd.GeoDataFrame, int]
        The clustered GeoDataFrame and the selected cluster
    """
    # Cluster the data. 
    kmeans = KMeans(n_clusters=cluster_nb, random_state=0).fit(r[column].values.reshape(-1,1))
    r["cluster"] = kmeans.labels_
    centres = kmeans.cluster_centers_

    if cluster_sense == "max":
        cluster = centres.argmax()
    elif cluster_sense == "min":
        cluster = centres.argmin()
    else:
        raise ValueError("Cluster sense must be either 'max' or 'min'. Others have not been implemented.")
    return r, cluster

def plot_clustered_map(
    config_name: str,
    period: pd.Series,
    event_nr: int,
    df: pd.DataFrame,
    cluster_sense: str,
    tech: str,
    norm: mpl.colors.Normalize,
    regions: gpd.GeoDataFrame,
    projection: ccrs.Projection,
    n: pypsa.Network,
    cluster_nb: int = 3,
    threshold: float = 0.9,
    use_anomalies: bool = False,
    averages: pd.DataFrame = None,
    ax: plt.Axes = None,
    save: bool = False,
    cmap: str = "coolwarm",
) -> plt.Axes:
    """Plot clustered data on a map.
    
    Parameters:
    -----------
    config_name: str
        Name of the configuration
    period: pd.Series
        Period to plot
    event_nr: int
        Event number
    df: pd.DataFrame
        Data to plot
    cluster_sense: str
        Whether to select the max or min cluster
    tech: str
        Technology to plot
    norm: matplotlib.colors.Normalize
        Color normalization
    regions: gpd.GeoDataFrame
        Regions to plot
    projection: cartopy.crs.Projection
        Projection to use
    n: pypsa.Network
        Network to plot
    cluster_nb: int
        Number of clusters
    threshold: float
        Threshold for the ratio of nodes in the cluster to all nodes in the region
    use_anomalies: bool
        Whether to use anomalies
    averages: pd.DataFrame
        Averages to compute anomalies
    ax: matplotlib.axes.Axes
        Axes to plot on
    save: bool
        Whether to save the hull
    cmap: str
        Colormap to use
        
    Returns:
    --------
    plt.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
    n.plot(ax=ax, bus_sizes=0, line_widths=0.5, link_widths=0.5)

    start = period.start
    end = period.end

    # Shift to access averages.
    shifted_start = f"1942{str(start)[4:]}" if start.month < 7 else f"1941{str(start)[4:]}"
    shifted_end = f"1942{str(end)[4:]}" if end.month < 7 else f"1941{str(end)[4:]}"

    # Set up GeoDataFrame.
    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    if use_anomalies:
        if averages is None:
            raise ValueError("Averages must be provided if anomalies are to be computed.")
        anomalies = df.loc[start:end].mean() - averages.loc[shifted_start:shifted_end].mean()
        if tech == "solar":
            anomalies.index = n.generators.loc[anomalies.index].bus.values
            anomalies = anomalies.groupby(anomalies.index).mean()
            r[tech] = anomalies
        elif tech == "load":
            anomalies /= averages.loc[shifted_start:shifted_end].mean()
            r[tech] = anomalies
        else:
            r[tech] = anomalies
    else:
        r[tech] = df.loc[start:end].mean()

    r.plot(ax=ax,
        column=tech,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    r, cluster = select_clusters(r, tech, cluster_nb, cluster_sense)
    sns.scatterplot(
        x="x",
        y="y",
        data=r,
        hue="cluster",
        palette="tab10",
        s=100,
        ax=ax,
        zorder=2,
        legend=False,
    )
    subregions, hull = draw_region(r, cluster, threshold, print_progress=False)
    if save:
        vertices = pd.DataFrame(hull.points[hull.vertices], columns=["x", "y"])
        if use_anomalies:
            vertices.to_csv(f"processing_data/{config_name}/maps/{tech}_anom/hull_{threshold}_event{event_nr}.csv")
        else:
            vertices.to_csv(f"processing_data/{config_name}/maps/{tech}/hull_{threshold}_event{event_nr}.csv")

    # Plot the convex hull.
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], "k-")
    return ax

def plot_clustered_wind(
    config_name: str,
    period: pd.Series,
    event_nr: int,
    df: pd.DataFrame,
    cluster_sense: str,
    tech: str,
    norm,
    regions: gpd.GeoDataFrame,
    offshore_regions: gpd.GeoDataFrame,
    projection,
    n: pypsa.Network,
    cluster_nb: int = 3,
    threshold: float = 0.9,
    averages: pd.DataFrame = None,
    ax = None,
    save: bool = False,
    cmap: str = "coolwarm",
) -> plt.Axes:
    """Plot clustered wind data.
    
    Note: this uses anomalies
    
    Parameters:
    -----------
    config_name: str
        Name of the configuration
    period: pd.Series
        Period to plot
    event_nr: int
        Event number
    df: pd.DataFrame
        Data to plot
    cluster_sense: str
        Whether to select the max or min cluster
    tech: str
        Technology to plot
    norm: matplotlib.colors.Normalize
        Color normalization
    regions: gpd.GeoDataFrame
        Regions to plot
    offshore_regions: gpd.GeoDataFrame
        Offshore regions to plot
    projection: cartopy.crs
        Projection to use
    n: pypsa.Network
        Network to plot
    cluster_nb: int
        Number of clusters
    threshold: float
        Threshold for the ratio of nodes in the cluster to all nodes in the region
    averages: pd.DataFrame
        Averages to compute anomalies
    ax: matplotlib.axes.Axes
        Axes to plot on
    save: bool
        Whether to save the hull
    cmap: str
        Colormap to use
        
    Returns:
    --------
    plt.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
    n.plot(ax=ax, bus_sizes=0, line_widths=0.5, link_widths=0.5)

    start = period.start
    end = period.end

    # Shift to access averages.
    shifted_start = f"1942{str(start)[4:]}" if start.month < 7 else f"1941{str(start)[4:]}"
    shifted_end = f"1942{str(end)[4:]}" if end.month < 7 else f"1941{str(end)[4:]}"

    # Set up GeoDataFrame.
    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    if averages is None:
        raise ValueError("Averages must be provided if anomalies are to be computed.")
    anomalies = df.loc[start:end].mean() - averages.loc[shifted_start:shifted_end].mean()

    # Need to separate onshore and offshore.
    onshore_anomalies = anomalies.filter(like="onwind", axis=0)
    offshore_anomalies = anomalies.filter(like="offwind", axis=0)
    onshore_anomalies.index = n.generators.loc[onshore_anomalies.index].bus.values
    offshore_anomalies.index = n.generators.loc[offshore_anomalies.index].bus.values
    # Merge ac, dc, float.
    offshore_anomalies = offshore_anomalies.groupby(offshore_anomalies.index).mean()

    # Add additional dataframe for offshore
    r_off = offshore_regions.set_index("name")
    r_off["x"], r_off["y"] = n.buses.x, n.buses.y
    r_off = gpd.geodataframe.GeoDataFrame(r_off, crs="EPSG:4326")
    r_off = r_off.to_crs(projection.proj4_init)

    # Separate on- and offshore wind.
    r["wind"] = onshore_anomalies
    r_off["offwind"] = offshore_anomalies

    r_off.plot(ax=ax,
        column="offwind",
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    r.plot(ax=ax,
        column=tech,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    r, cluster = select_clusters(r, tech, cluster_nb, cluster_sense)
    sns.scatterplot(
        x="x",
        y="y",
        data=r,
        hue="cluster",
        palette="tab10",
        s=100,
        ax=ax,
        zorder=2,
        legend=False,
    )
    subregions, hull = draw_region(r, cluster, threshold, print_progress=False)
    if save:
        vertices = pd.DataFrame(hull.points[hull.vertices], columns=["x", "y"])
        vertices.to_csv(f"processing_data/{config_name}/maps/{tech}_anom/hull_{threshold}_event{event_nr}.csv")

    # Plot the convex hull.
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], "k-")
    return ax

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
    '''Plot affected areas on a map based on pre-computed hulls.
    
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
        
    Returns:
    --------
    list
        Legend elements for the plot
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": projection})
    n.plot(ax=ax, bus_sizes=0, bus_colors="black", line_widths=0, link_widths=0, link_colors="black", line_colors="black",color_geomap=True)

    start = period.start
    end = period.end

    r = regions.set_index("name")
    r["x"], r["y"] = n.buses.x, n.buses.y
    r = gpd.geodataframe.GeoDataFrame(r, crs="EPSG:4326")
    r = r.to_crs(projection.proj4_init)

    r[fill_tech] = fill_df.loc[start:end].mean()

    r.plot(ax=ax,
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
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.05, aspect=25, shrink=0.75)
    
    ticks = [0, 0.25, 0.5, 0.75, 1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0%}" for t in ticks], fontsize=6)
    
    legend_elements = []
    hatches = [None, None, None, "x", "/", "o", ".", "*", "O", "-"]

    for hull, tech, colour, pretty_name, hatch in zip(hulls, techs, colours, pretty_names, hatches):
        # For now, only edges, no filling.
        hull_transformed = projection.transform_points(ccrs.PlateCarree(), hull.points[hull.vertices][:,0], hull.points[hull.vertices][:,1])
        patch = Polygon(xy=hull_transformed[:, :2], closed=True, ec=colour, fill=False, lw=1, zorder=2)
        legend_elements.append(Patch(ec = colour, fill = False, lw = 1, label=pretty_name))
        ax.add_patch(patch)
    return legend_elements
def grid_wind(
    config_name: str,
    periods: pd.DataFrame,
    df: pd.DataFrame,
    cluster_sense: str,
    tech: str,
    norm: mpl.colors.Normalize,
    regions: gpd.GeoDataFrame,
    offshore_regions: gpd.GeoDataFrame,
    projection: ccrs.Projection,
    n: pypsa.Network,
    cluster_nb: int = 3,
    threshold: float = 0.9,
    averages: pd.DataFrame = None,
    save: bool = False,
    cmap: str = "coolwarm",
) -> None:
    """Note: this uses anomalies"""
    nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1

    fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

    for i, period in enumerate(periods.iterrows()):
        ax = axs[i // 3, i % 3]
        
        plot_clustered_wind(
            config_name,
            periods.loc[i],
            i,
            df,
            cluster_sense,
            tech,
            norm,
            regions = regions,
            offshore_regions = offshore_regions,
            projection = projection,
            n = n,
            cluster_nb = cluster_nb,
            threshold = threshold,
            averages = averages,
            ax = ax,
            save = save,
            cmap = cmap,
        )
        ax.set_title(f"Event {i}: {periods.loc[i, "start"]} - {periods.loc[i, "end"]}")
    plt.tight_layout()
    plt.show()
    plt.close()

    if save:
        fig.savefig(f"processing_data/{config_name}/maps/{tech}_anom/clustered_{tech}_anom_{cluster_sense}.pdf", bbox_inches="tight")


def grid_maps(
    config_name: str,
    periods: pd.DataFrame,
    df: pd.DataFrame,
    cluster_sense: str,
    tech: str,
    norm: mpl.colors.Normalize,
    regions: gpd.GeoDataFrame,
    projection: ccrs.Projection,
    n: pypsa.Network,
    cluster_nb: int = 3,
    threshold: float = 0.9,
    use_anomalies: bool = False,
    averages: pd.DataFrame = None,
    save: bool = False,
    cmap: str = "coolwarm",
) -> None:
    """Create a grid of maps showing clustered data.
    
    Parameters:
    -----------
    config_name: str
        Name of the configuration
    periods: pd.DataFrame
        DataFrame containing the periods to plot
    df: pd.DataFrame
        Data to plot
    cluster_sense: str
        Whether to select the max or min cluster
    tech: str
        Technology to plot
    norm: mpl.colors.Normalize
        Color normalization
    regions: gpd.GeoDataFrame
        Regions to plot
    projection: ccrs.Projection
        Projection to use
    n: pypsa.Network
        Network to plot
    cluster_nb: int, default 3
        Number of clusters
    threshold: float, default 0.9
        Threshold for the ratio of nodes in the cluster to all nodes in the region
    use_anomalies: bool, default False
        Whether to use anomalies
    averages: pd.DataFrame, optional
        Averages to compute anomalies
    save: bool, default False
        Whether to save the figure
    cmap: str, default "coolwarm"
        Colormap to use
    """
    nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1
    fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})
    for i, period in enumerate(periods.iterrows()):
        ax = axs[i // 3, i % 3]
        
        plot_clustered_map(
            config_name,
            periods.loc[i],
            i,
            df,
            cluster_sense,
            tech,
            norm,
            regions = regions,
            projection = projection,
            n = n,
            cluster_nb = cluster_nb,
            threshold = threshold,
            use_anomalies = use_anomalies,
            averages = averages,
            ax = ax,
            save = save,
            cmap = cmap,
        )
        ax.set_title(f"{periods.loc[i, 'start']} - {periods.loc[i, 'end']}")
    plt.tight_layout()
    plt.show()
    plt.close()

    if save:
        if use_anomalies:
            fig.savefig(f"processing_data/{config_name}/maps/{tech}_anom/clustered_{tech}_anom_{cluster_sense}.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"processing_data/{config_name}/maps/{tech}/clustered_{tech}_{cluster_sense}.pdf", bbox_inches="tight")

def grid_affected_areas(
    config_name: str,
    periods: pd.DataFrame,
    hulls_collection: dict[str, list[ConvexHull]],
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
    save: bool = False,
) -> None:
    """Create a grid of maps showing affected areas for multiple periods.
    
    Parameters:
    -----------
    config_name: str
        Name of the configuration
    periods: pd.DataFrame
        DataFrame containing the periods to plot
    hulls_collection: dict[str, list[ConvexHull]]
        Dictionary mapping technologies to lists of pre-computed hulls
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
    save: bool, default False
        Whether to save the figure
    """
    nb_rows = len(periods) // 3 if len(periods) % 3 == 0 else len(periods) // 3 + 1
    fig, axs = plt.subplots(nb_rows, 3, figsize=(18, 6 * nb_rows), subplot_kw={"projection": projection})

    for (i, period) in enumerate(periods.iterrows()):
        ax = axs[i // 3, i % 3]
        hulls_period = [hulls_collection[tech][i] for tech in techs]
        
        legend_elements = plot_affected_areas(
            config_name,
            period[1],
            i,
            hulls_period,
            techs,
            pretty_names,
            colours,
            fill_df,
            fill_tech,
            fill_norm,
            fill_cmap,
            regions,
            n,
            projection,
            ax = ax,
            save = save,
        )
        ax.set_title(f"Event {i}: {periods.loc[i, "start"]} - {periods.loc[i, "end"]}")
    
    # Add legend to the last row.
    ax_l = axs[-1, 0]
    ax_l.legend(handles = legend_elements, ncols = 4, bbox_to_anchor=(2, -0.1), loc='upper center')

    if save:
        fig.savefig(f"processing_data/{config_name}/system_maps/affected_areas.pdf", bbox_inches="tight")

    