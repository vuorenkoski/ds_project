import numpy as np
import pandas as pd
from tslearn.metrics import soft_dtw
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.manifold import TSNE
import geopandas as gpd
import os

df1 = pd.read_csv('countries.csv')
df2 = pd.read_csv('data.csv')
df3 = pd.read_csv('features.csv')
data = df2.copy()

def cluster_into_n(df, max_clusters, show_graph=False):
    def clustered_dfs_gen(clustered_dfs):
        for i in range(len(clustered_dfs)):
            yield clustered_dfs[i]
    # missing data extrapolation
    df = df.interpolate('linear').fillna(0)
    # cluster and pick n clusters with best results
    model = TimeSeriesKMeans(n_clusters=max_clusters*2, metric="euclidean", max_iter=10)
    model.fit(df.values)
    df['cluster'] = model.labels_
    n_clusters = min(len(df.cluster.unique()), max_clusters)
    clustered_dfs = [df.loc[df.cluster == i] for i in range(n_clusters)]
    clustered_dfs = sorted([(len(df), idx) for idx, df in enumerate(clustered_dfs)])[::-1][:n_clusters]
    clustered_dfs = [df.loc[df.cluster == tup[1]] for tup in clustered_dfs]
    # show graph if necessary
    if show_graph:
        gen = clustered_dfs_gen(clustered_dfs)
        fig, axs = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                try:
                    cluster = next(gen)
                    cluster.iloc[:, :-1].T.plot(ax=axs[i][j])   
                    ylim = (df.max().max(), df.min().min())
                    axs[i][j].set_ylim(ylim[0], ylim[1])
                except:
                    pass
        plt.show()
    return clustered_dfs

def EDA(mode):
    def cluster_vaccination_rate():
        def preprocessing():
            # load data
            df = pd.read_csv('owid-covid-data.csv')
            df = df[['people_vaccinated_per_hundred', 'iso_code', 'date']]
            df['month'] = df.date.str[0:7]
            # process into monthly data
            lst = []
            for code in df.iso_code.unique():
                partial_df = df.loc[df.iso_code == code]
                # max because values are accumulative, interpolate to fill missing values in between, fill rest with 0s
                partial_df = partial_df.groupby('month').max().interpolate('linear').fillna(0).reset_index()
                result_df = partial_df.pivot('month', 'iso_code', 'people_vaccinated_per_hundred').T
                lst.append(result_df)
            processed_df = pd.concat(lst)
            processed_df = processed_df.reindex(sorted(processed_df.columns), axis=1)
            processed_df = processed_df.interpolate('linear').fillna(0)
            return processed_df

        processed_df = preprocessing()
        result = cluster_into_n(processed_df, 9, show_graph=False)

        # only leave clusters with more than 3 entries, clusters all extra entries into 1 cluster
        clusters = [i for i in result if len(i) > 3]
        reindexed_clusters = []
        for idx, df in enumerate(clusters):
            temp_df = df.copy()
            temp_df.cluster = idx+1
            reindexed_clusters.append(temp_df)
        if len(reindexed_clusters) != 9:
            extra_cluster = pd.concat([i for i in result if len(i) <= 3])
            extra_cluster.cluster = len(reindexed_clusters) + 1
            reindexed_clusters.append(extra_cluster)
        result_df = pd.concat(reindexed_clusters)
        return result_df.cluster

    def cluster_yearly_data():
        lst = []
        for i in [1, 2, 3, 4, 5]:
            pivoted_df = data.loc[data.feature_id == i].pivot('country_code', 'year', 'value')
            result = cluster_into_n(pivoted_df, 9, show_graph=False)
            clusters = [i for i in result if len(i) > 3]
            reindexed_clusters = []
            for idx, df in enumerate(clusters):
                temp_df = df.copy()
                temp_df.cluster = idx+1
                reindexed_clusters.append(temp_df)
            if len(reindexed_clusters) != 9:
                extra_cluster = pd.concat([i for i in result if len(i) <= 3])
                extra_cluster.cluster = len(reindexed_clusters) + 1
                reindexed_clusters.append(extra_cluster)
            result_df = pd.concat(reindexed_clusters)
            lst.append(result_df.cluster)
        processed_features = pd.concat(lst, axis=1)
        return processed_features

    # append cluster data from monthly vaccination data 
    processed_features = cluster_yearly_data()
    vaccination_data = cluster_vaccination_rate().rename('vaccination')
    processed_features.columns = ['gdp', 'tourism', 'unemployment', 'gdp_growth', 'debt_rate']
    processed_features = processed_features.join(vaccination_data)

    # final clustering : missing data extrapolation
    processed_features = processed_features.fillna(0)

    if mode == 'covid':
        processed_features = processed_features[['tourism', 'vaccination']]
    elif mode == 'economy':
        processed_features = processed_features[['gdp', 'gdp_growth', 'unemployment', 'debt_rate']]
    elif mode == 'all':
        pass

    # TODO cluster using k-modes clustering 
    model = KModes(n_clusters=9, init = "Huang", n_init = 1, verbose=1)
    clustering_result = model.fit_predict(processed_features)
    processed_features['cluster'] = clustering_result
    target_df = processed_features.astype(float).iloc[:, :-1]
    return processed_features

    # visualize after dimensionality reduction 
    '''
    model = TSNE(random_state=0)
    tsne_result = model.fit_transform(target_df.values)
    tsne_data = pd.DataFrame(tsne_result, columns=['x', 'y'], index=target_df.index)
    tsne_data['cluster'] = clustering_result
    '''

    '''
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.clf()
    plt.cla()
    for i in range(9):
        partial_df = tsne_data.loc[tsne_data.cluster == i]
        plt.scatter(partial_df.x, partial_df.y, color=colors[i])
    plt.show()
    #tsne_result = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(target_df.values)
    breakpoint()
    '''

def visualize(clustered_df, individual=False):
    # gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).set_index('iso_a3')
    world = world.join(clustered_df.cluster)[['name', 'geometry', 'cluster']]
    if individual:    
        world.plot(column='cluster', cmap='Accent')
        plt.show()
    else:
        individual_dfs = []
        for cluster in world.dropna().cluster.unique():
            partial_df = world.copy()
            target_df = partial_df.loc[partial_df.cluster == cluster]
            non_target_df = partial_df.loc[partial_df.cluster != cluster]
            extra_df = partial_df.loc[partial_df.cluster.isnull()]

            target_df.cluster = 0
            non_target_df.cluster = 1
            result_df = pd.concat([target_df, non_target_df, extra_df])
            individual_dfs.append(result_df)

        def gen(lst):
            for i in lst:
                yield i

        fig, axs = plt.subplots(3, 3)
        df_gen = gen(individual_dfs)
        for i in range(3):
            for j in range(3):
                next(df_gen).plot(column='cluster', ax=axs[i][j], cmap='Accent')
        plt.show()

if __name__ == "__main__":
    for mode in ['covid', 'economy', 'all']:
        if 'clustered_df_{}.csv'.format(mode) not in os.listdir(os.getcwd()):
            clustered_df = EDA(mode=mode)
            clustered_df.to_csv('clustered_df_{}.csv'.format(mode))
        else:
            clustered_df = pd.read_csv('clustered_df_{}.csv'.format(mode)).set_index('country_code')
        visualize(clustered_df, individual=True)
        #visualize(clustered_df, individual=False)