from sklearn.metrics import jaccard_score as jacs
from sklearn.metrics.cluster import adjusted_rand_score as ars
from scipy.stats import false_discovery_control as fdrc
from os import path
from multiprocessing import Pool
from tqdm import tqdm
from itertools import combinations
import networkx as nx
import pandas as pd
import numpy as np
import orca
import gc
from time import sleep

nx.config.warnings_to_ignore.add("cache")

PERMS_DIR = "permutations"
DISEASES = ["BD", "MDD", "SCZ", "Healthy"]
N_PERMUTATIONS = 1000
THRESHOLD = 0.3
OUTPUT_DIR = "network_results"

def main():
    
    combs = combinations(DISEASES, 2)

    for (disease1, disease2) in combs:

        inputDir = path.join(PERMS_DIR, disease1+disease2, "cor")
        df1 = pd.read_csv(path.join(inputDir, f"{disease1}.tsv"), sep="\t", header=0, index_col=0)
        df2 = pd.read_csv(path.join(inputDir, f"{disease2}.tsv"), sep="\t", header=0, index_col=0)
        
        taxa = list(df1.columns)
        
        graph1 = buildNet(df1)
        graph2 = buildNet(df2)

        refMeasures, refCentralities = difference(graph1, graph2, taxa)
        refCentralities.to_csv(path.join(OUTPUT_DIR, f"{disease1+disease2}-cen.csv"))
        refMeasures.to_csv(path.join(OUTPUT_DIR, f"{disease1+disease2}-measures.csv"))
        
        pvalMeasures = pd.DataFrame(0, index=refMeasures.index, columns=refMeasures.columns)
        pvalCentralities = pd.DataFrame(0, index=refCentralities.index, columns=refCentralities.columns)     
       
        
        with Pool() as pool:
            results = pool.starmap(processPermutation,
                                   tqdm([(i, inputDir, taxa) for i in range(N_PERMUTATIONS)], 
                                        desc=f"{disease1+disease2} p-values"))


        for diffMeasures, diffCentralities in results:
            pvalMeasures += (diffMeasures >= refMeasures)
            pvalCentralities += (diffCentralities >= refCentralities)
        
        pvalMeasures = pvalMeasures / N_PERMUTATIONS
        pvalMeasures.iloc[:] = fdrc(pvalMeasures.values.flatten()).reshape(pvalMeasures.shape)
        pvalMeasures.to_csv(path.join(OUTPUT_DIR, f"{disease1+disease2}_measure_pvalues.csv"))

        pvalCentralities = pvalCentralities / N_PERMUTATIONS
        pvalCentralities.iloc[:] = fdrc(pvalCentralities.values.flatten()).reshape(pvalCentralities.shape)
        pvalCentralities.to_csv(path.join(OUTPUT_DIR, f"{disease1+disease2}_centrality_pvalues.csv"))
        sleep(60)



def buildNet(cordf):
   
    cor_values = cordf.values
    mask = np.triu(np.ones(cor_values.shape, dtype=bool))
    cor_values[~mask] = 0
    
    cor_values[np.abs(cor_values) <= THRESHOLD] = 0
    
    distance = np.sqrt(0.5*(1 - cor_values))
    distance[cor_values == 0] = np.inf
    similarity = 1 - distance
    
    rows, cols = np.where(mask)
    links = pd.DataFrame({
        'OTU1': cordf.index[rows],
        'OTU2': cordf.columns[cols],
        "distance": distance[rows, cols],
        'similarity': similarity[rows, cols]
    })
    links = links[(links['OTU1'] != links['OTU2']) & (links["similarity"] > 0)]
    
    graph = nx.from_pandas_edgelist(links, 'OTU1', 'OTU2', edge_attr=["similarity", "distance"])
    return graph



def intraNetMeasures(graph, taxa):

    measures = {}

    centralities = pd.DataFrame(index=taxa)
    centralities["degree centrality"] = nx.degree_centrality(graph)
    centralities["eigenvector centrality"] = nx.eigenvector_centrality(graph, weight="similarity")
    centralities["closeness centrality"] = nx.closeness_centrality(graph, distance="distance")
    centralities["betweenness centrality"] = nx.betweenness_centrality(graph, weight="similarity")
    centralities = centralities.fillna(0)

    lcc = graph.subgraph(max(nx.connected_components(graph), key=len))

    hubs = nx.pagerank(graph, weight="similarity")
    hubs = np.array(sorted(hubs.items(), key=lambda item: item[1], reverse=True))[:10, 0]
    
    measures["lccRatio"] = lcc.number_of_nodes()/graph.number_of_nodes()
    measures["clustering"] = nx.average_clustering(graph, weight="similarity")

    communities = nx.community.louvain_communities(graph, weight="similarity")
    measures["modularity"] = nx.community.modularity(graph, communities, weight="similarity")

    measures["assortativity"] = nx.degree_assortativity_coefficient(graph, weight="similarity")
    measures["PosEdgeRatio"] = np.mean(np.array([s for u, v, s in graph.edges(data="similarity")]) > (1-np.sqrt(0.5)))
    measures["graphDensity"] = nx.density(graph)
    measures["lccDensity"] = nx.density(lcc)
    measures["diameter"] = nx.diameter(lcc, weight="distance")
    measures["radius"] = nx.radius(lcc, weight="distance")

    eigenvalues = np.linalg.eigvals(nx.to_numpy_array(lcc, weight="similarity"))
    measures["naturalConnectivity"] = np.log(np.mean(np.exp(eigenvalues))).real
    measures["nodeConnectivity"] = nx.node_connectivity(lcc) 
    measures["edgeConnectivity"] = nx.edge_connectivity(lcc) 

    return pd.DataFrame(measures, index=[0]), centralities, communities, hubs



def difference(graph1, graph2, taxa):

    m1 , cen1, com1, h1 = intraNetMeasures(graph1, taxa)
    m2, cen2, com2, h2 = intraNetMeasures(graph2, taxa)
    
    diffm = np.abs(m2 - m1)
    diffcen = np.abs(cen2 - cen1)

    diffm["Jaccard"] = np.abs(jaccard(h1, h2, taxa))
    diffm["Rand"] = aRand(com1, com2, taxa)
    diffm["GCD"] = GCD(graph1, graph2)

    return diffm, diffcen



def GCD(graph1, graph2):

    gcv1 = pd.DataFrame(orca.orbit_counts("node", 4, graph1)).drop(columns=[3, 12, 13, 14])
    gcv1.loc[len(gcv1)] = 1
    gcm1 = gcv1.corr("spearman")
    gcm1 = np.array(gcm1)[np.triu_indices_from(gcm1, k=1)].flatten()

    gcv2 = pd.DataFrame(orca.orbit_counts("node", 4, graph2)).drop(columns=[3, 12, 13, 14])
    gcv2.loc[len(gcv2)] = 1
    gcm2 = gcv2.corr("spearman")
    gcm2 = np.array(gcm2)[np.triu_indices_from(gcm2, k=1)].flatten()

    return np.linalg.norm(gcm2 - gcm1)



def jaccard(hubs1, hubs2, taxa):
    
    pind = 10/len(taxa)
    class1 = np.array([int(otu in hubs1) for otu in taxa])
    class2 = np.array([int(otu in hubs2) for otu in taxa])
    return jacs(class1, class2) - (pind**2)/(2*pind - pind**2)



def aRand(communities1, communities2, taxa):

    class1 = []
    for otu in taxa:
        for i, community in enumerate(communities1):
            if otu in community:
                class1.append(i)
                break
        else:
            class1.append(-1)

    class2 = []
    for otu in taxa:
        for i, community in enumerate(communities2):
            if otu in community:
                class2.append(i)
                break
        else:
            class2.append(-1)

    return ars(np.array(class1), np.array(class2))



def processPermutation(i, inputDir, taxa):

    df1 = pd.read_csv(path.join(inputDir, f"{i}_1.tsv"), sep="\t", header=0, index_col=0)
    df2 = pd.read_csv(path.join(inputDir, f"{i}_2.tsv"), sep="\t", header=0, index_col=0)
    
    graph1 = buildNet(df1)
    graph2 = buildNet(df2)
    
    diffMeasures, diffCentralities = difference(graph1, graph2, taxa)
    
    del df1, df2, graph1, graph2
    gc.collect()
    
    return diffMeasures, diffCentralities



if __name__ == "__main__":
    main()
    