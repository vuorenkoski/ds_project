import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

def remove_outliers_df(df):
    def remove_outliers_srs(data):
        srs = data.copy()
        q1 = np.percentile(srs, 25)
        q3 = np.percentile(srs, 75)
        iqr = q3-q1

        minimum = q1 - 1.5 * iqr
        maximum = q3 + 1.5 * iqr
        srs.loc[srs < minimum] = np.NaN
        srs.loc[srs > maximum] = np.NaN
        return srs
               
    data = df.T.copy()
    for colname in data.keys():
        data[colname] = remove_outliers_srs(data[colname])
    return data.T


def cluster_into_9(input_df, mode):
    '''
    for debt data, cluster into 12 then pick 9 clusters with the most elements
    '''
    df = input_df.copy()
    if mode=='debt':
        max_clusters=12
        take_clusters=9
        ylim = (0, 150)
    elif mode=='gdp':
        max_clusters=18
        take_clusters=9
        ylim = (0, 1000)
    elif mode=='gdp_growth':
        max_clusters=18
        take_clusters=9
        ylim = (0, 3)

    from tslearn.metrics import soft_dtw
    from tslearn.clustering import TimeSeriesKMeans
    model = TimeSeriesKMeans(n_clusters=max_clusters, metric="dtw", max_iter=10)
    model.fit(df.values)
    df['cluster'] = model.labels_
    
    clustered_dfs = [df.loc[df.cluster == i] for i in range(max_clusters)]
    clustered_dfs = sorted([(len(df), idx) for idx, df in enumerate(clustered_dfs)])[::-1][:take_clusters]
    clustered_dfs = [df.loc[df.cluster == tup[1]] for tup in clustered_dfs]
    def clustered_dfs_gen(clustered_dfs):
        for i in range(len(clustered_dfs)):
            yield clustered_dfs[i]
    gen = clustered_dfs_gen(clustered_dfs)
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            cluster = next(gen)
            if len(cluster)==0:
                pass
            else:
                cluster.iloc[:, :-1].T.plot(ax=axs[i][j])   
                axs[i][j].set_ylim(ylim[0], ylim[1])
    plt.show()

# load data
gdp = pd.read_excel('Seungwon/imf_gdp_formatted.xlsx').rename(columns={'Unnamed: 0':'Country'}).set_index('Country').replace('...', np.NaN).interpolate('linear').fillna(0)
debt = pd.read_excel('Seungwon/imf_debt.xlsx').rename(columns={'Central Government Debt (Percent of GDP)':'Country'}).set_index('Country').replace('...', np.NaN).iloc[:-2, -10:]
for col in debt.keys():
    debt[col] = pd.to_numeric(debt[col], errors='coerce')

# missing data extrapolation
debt = debt.interpolate('linear').fillna(0)

# calculate gdp growth data
gdp_growth_df = gdp.copy()
for i in reversed(range(len(gdp.keys()))):
    gdp_growth_df.iloc[:, i] = gdp_growth_df.iloc[:, i] / (gdp_growth_df.iloc[:, i-1] + 1e-9)
processed_gdp_df = remove_outliers_df(gdp).interpolate('linear')
processed_gdp_growth_df = remove_outliers_df(gdp_growth_df).interpolate('linear')

# show time series clustering results
cluster_into_9(debt, 'debt')
#cluster_into_9(processed_gdp_df, 'gdp')
#cluster_into_9(processed_gdp_growth_df, 'gdp_growth')

# save processed gdp growth data as csv
processed_gdp_growth_df.to_csv('Seungwon/processed_gdp_growth.csv', index=False)

'''
# legacy code
q4_2019_gdp = processed_gdp_df['2019Q4'].rename('gdp')
q4_2019_gdp_growth = processed_gdp_growth_df['2019Q4'].rename('gdp_growth')
debt_2019 = debt[2019].rename('debt')
debt_2019 = debt_2019.loc[debt_2019.index.isin(q4_2019_gdp.index)]
q4_2019_gdp = q4_2019_gdp.loc[q4_2019_gdp.index.isin(debt_2019.index)]
q4_2019_gdp_growth = q4_2019_gdp_growth.loc[q4_2019_gdp_growth.index.isin(debt_2019.index)]

df = pd.concat([debt_2019, q4_2019_gdp, q4_2019_gdp_growth], axis=1)


from sklearn.manifold import TSNE
n_components = 2
model = TSNE(n_components=n_components, n_iter=10000, n_iter_without_progress=10000)
results = model.fit_transform(df.values)

plt.figure(figsize=(10,10))
plt.xlim(results[:, 0].min(), results[:, 0].max())
plt.ylim(results[:, 1].min(), results[:, 1].max())

for i in range(len(results)):
    plt.text(results[i, 0], results[i,1], str(debt_2019.index[i]), fontdict={'weight': 'bold', 'size':9})
plt.show()
'''