# global imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns

from graphdatascience import GraphDataScience
from matplotlib.ticker import FormatStrFormatter
import sys

# local imports
from AnalysisManager import AnalysisManager
from NodeMetrics import NodeMetrics
from GraphMetrics import GraphMetrics


def main():

    usr = sys.argv[1]
    pwd = sys.argv[2]

    # Neo4j: classes instances for analyzers
    manager = AnalysisManager(usr, pwd, "../data", outputs_base_location="../")
    connector = GraphDataScience("bolt://localhost:7687", auth=(usr, pwd))
    node_metrics = NodeMetrics(connector, "paysim", "node", "TRANSACTION")
    graph_metrics = GraphMetrics(connector, "paysim", "node", "TRANSACTION")

    # NetworkX: objects for functions based on NetworkX
    edgelist = pd.read_csv("../data/paysim_3_timesteps.csv")
    graphtype = nx.DiGraph()
    graph = nx.from_pandas_edgelist(edgelist,
                                    source="nameOrig",
                                    target="nameDest",
                                    create_using=graphtype)

# Note on shortest paths
# The shortest paths are determined using Networkx class static method shortest_path_length()
# which returns pairs of node ID and a dictionary containing all its neighbours IDs and path lengths to them.
# In this case, the information is stored in a multidimensional array of heterogeneous sizes of elements
# which store the origin node ID, destination node IDs and path lengths.
def getAllShortestPaths(graph):
    all_shortest_paths = []
    for (i,j) in nx.shortest_path_length(graph):
        all_shortest_paths.append([i, list(j.keys()), np.array(list(j.values()))])
    return all_shortest_paths


def getMeanOfShortestPaths(shortest_paths):
    summed_lengths = 0
    number_of_paths = 0
    for i in range(len(shortest_paths)):
        summed_lengths += (shortest_paths[i][2].sum())
        number_of_paths += sum(shortest_paths[i][2] != 0)

    return summed_lengths / number_of_paths


def getMedianOfShortestPaths(shortest_paths):
    all_paths_lengths = []
    for i in range(len(shortest_paths)):
        for path_length in shortest_paths[i][2]:
            all_paths_lengths.append(path_length)
    all_paths_lengths = pd.Series(np.array(all_paths_lengths))
    return all_paths_lengths.median()


def getDiameter(shortest_paths):
    all_paths_lengths = []
    for i in range(len(shortest_paths)):
        for path_length in shortest_paths[i][2]:
            all_paths_lengths.append(path_length)
    all_paths_lengths = pd.Series(np.array(all_paths_lengths))
    return all_paths_lengths.max()


def getDegreeDistribution(connector, nodes_and_classes):
    nodes = nodes_and_classes["id"]
    degree_df = None
    index = 0
    for node in nodes:
        index += 1
        print(f"Progress: {index}/{len(nodes)}", end='\r')
        results_in_degree = connector.run_cypher("""
                        MATCH (n:node)<-[r:TRANSACTION]-(m:node)
                        WHERE n.id = '%s'
                        RETURN n.id AS id, count(m) AS in_degree
                        """ % node)
        results_out_degree = connector.run_cypher("""
                        MATCH (n:node)-[r:TRANSACTION]->(m:node)
                        WHERE n.id = '%s'
                        RETURN n.id AS id, count(m) AS out_degree
                        """ % node)

        merged = pd.merge(results_in_degree, results_out_degree, on="id", how="outer")

        if degree_df is None:
            degree_df = merged
        else:
            degree_df = pd.concat([degree_df, merged], axis=0)

    return degree_df.fillna(0)


# ## Large-scale Network Properties

def getLargeScaleProperties(connector, node_metrics, graph_metrics):

    nodes_classes = node_metrics.getNodesClasses()
    degree_distribution_df = getDegreeDistribution(connector=connector, nodes_and_classes=nodes_classes)
    column_names = ["Type", "n", "m", "c", "S", "l", "alpha_in", "alpha_out", "C"]
    properties = ["Directed"]

    # nodes
    n = graph_metrics.getGraphSize()
    properties.append(n)

    # edges
    m = graph_metrics.getNumberOfEdges().values[0][0]
    properties.append(m)

    # Mean degree
    c = 2 * m / n
    properties.append(c)

    # Fraction of nodes in the giant component (the largest component)
    fractions = pd.DataFrame(graph_metrics.getFractionsWeaklyConnectedComponents())
    fractions = fractions.reset_index()
    fractions.columns = ["componentId", "fraction_of_nodes"]
    S = fractions[fractions["fraction_of_nodes"] == fractions["fraction_of_nodes"].max()]["fraction_of_nodes"][0]
    properties.append(S)

    # Mean distance between connected node pairs
    l = getMeanOfShortestPaths(getAllShortestPaths(graph=graph))
    properties.append(l)

    # Exponent alpha
    x1 = degree_distribution_df["in_degree"].values
    x2 = degree_distribution_df["out_degree"].values
    data1 = powerlaw.Fit(x1)
    data2 = powerlaw.Fit(x2)
    properties.append(data1.alpha)
    properties.append(data2.alpha)



    # Mean clustering coefficient
    local_clustering_coefficients = node_metrics.getClusteringCoefficient()
    C = local_clustering_coefficients["localClusteringCoefficient"].mean()
    properties.append(C)

    large_scale_structure_df = pd.DataFrame([properties])
    large_scale_structure_df.columns = column_names
    large_scale_structure_df.to_csv("../graph_large_scale_properties.csv")
    return large_scale_structure_df


# ### Additional properties

# Density

# In[21]:


rho = (2 * m) / (n * (n - 1))
print(rho)


# Number of strongly connected components

# In[22]:


len(graph_metrics.getFractionsStronglyConnectedComponents())


# Number of weakly connected components

# In[23]:


len(graph_metrics.getFractionsWeaklyConnectedComponents())


# Graph's diameter

# In[24]:


path_lengths = {}
all_shortest_paths = getAllShortestPaths(graph)
for i in range(len(all_shortest_paths)):
    for path in all_shortest_paths[i][2]:
        if path in path_lengths.keys():
            path_lengths[path] += 1
        else:
            path_lengths[path] = 1


# In[25]:


path_lengths


# In[26]:


getDiameter(getAllShortestPaths(graph=graph))


# ## Centrality Metrics Analysis

# #### Class imbalance

# In[27]:


nodes_classes["class"].value_counts()


# ### Degree Distribution

# In[28]:


degree_distribution_df.head()


# In[149]:


def plotInDegreeDistribution(x, y, color, title_postfix=""):
    plt.figure(figsize=(8,7))
    sns.barplot(x=x, y=y, color=color)
    plt.xlabel("in-degree")
    plt.ylabel("count")
    plt.title(f"In-degree distribution{title_postfix}", fontsize=16)
    plt.show()

def plotOutDegreeDistribution(x, y, color, title_postfix=""):
    plt.figure(figsize=(8,7))
    sns.barplot(x=x, y=y, color=color)
    plt.xlabel("out-degree")
    plt.ylabel("count")
    plt.title(f"Out-degree distribution{title_postfix}", fontsize=16)
    plt.show()


# In[150]:


plotInDegreeDistribution(x=degree_distribution_df["in_degree"].value_counts().index, y=degree_distribution_df["in_degree"].value_counts(), color="lightseagreen")


# In[151]:


plotOutDegreeDistribution(x=degree_distribution_df["out_degree"].value_counts().index, y=degree_distribution_df["out_degree"].value_counts(), color="orange")


# In[129]:


degree_distribution_and_class = pd.merge(nodes_classes, degree_distribution_df, on="id")


# In[152]:


x = degree_distribution_and_class[degree_distribution_and_class["class"] == 1]["in_degree"].value_counts().index
y = degree_distribution_and_class[degree_distribution_and_class["class"] == 1]["in_degree"].value_counts()
plotInDegreeDistribution(x=x, y=y, title_postfix=": fraudulent nodes", color="crimson")


# In[153]:


x = degree_distribution_and_class[degree_distribution_and_class["class"] == 0]["in_degree"].value_counts().index
y = degree_distribution_and_class[degree_distribution_and_class["class"] == 0]["in_degree"].value_counts()
plotInDegreeDistribution(x=x, y=y, title_postfix=": non-fraudulent nodes", color="lightseagreen")


# In[154]:


x = degree_distribution_and_class[degree_distribution_and_class["class"] == 1]["out_degree"].value_counts().index
y = degree_distribution_and_class[degree_distribution_and_class["class"] == 1]["out_degree"].value_counts()
plotOutDegreeDistribution(x=x, y=y, title_postfix=": fraudulent nodes", color="crimson")


# In[155]:


x = degree_distribution_and_class[degree_distribution_and_class["class"] == 0]["out_degree"].value_counts().index
y = degree_distribution_and_class[degree_distribution_and_class["class"] == 0]["out_degree"].value_counts()
plotOutDegreeDistribution(x=x, y=y, title_postfix=": non-fraudulent transactions", color="lightseagreen")


# In[37]:


degree_distribution_and_class[degree_distribution_and_class["class"] == 0]["in_degree"].value_counts()


# In[38]:


degree_distribution_and_class[degree_distribution_and_class["class"] == 1]["in_degree"].value_counts()


# ### Pagerank scores distribution

# In[39]:


def plotPageRankDistribution(subset, color, title_postfix=""):
    subset = subset.value_counts()
    plt.figure(figsize=(8,7))
    sns.barplot(x=subset, y=subset.index, color=color)
    plt.xlabel("pagerank score")
    plt.xticks(rotation=45)
    plt.ylabel("count")
    plt.title(f"Pagerank scores distribution{title_postfix}")
    plt.show()


# In[40]:


pagerank_scores = node_metrics.getPageRankScores().round(decimals=3)
pagerank_scores_classes = pd.merge(pagerank_scores, nodes_classes, on="id")


# In[41]:


manager.getCentralityMetricsDataFrame("PageRankScore")


# In[157]:


plt.figure(figsize=(8,7))
sns.barplot(x=pagerank_scores_classes["PageRankScore"].value_counts().index, y=pagerank_scores_classes["PageRankScore"].value_counts(), color="lightseagreen")
plt.xlabel("pagerank score")
plt.xticks(rotation=45)
plt.ylabel("count")
plt.title(f"Pagerank scores distribution")
plt.show()


# In[158]:


subset = pagerank_scores_classes[pagerank_scores_classes["class"] == 1]["PageRankScore"]
plt.figure(figsize=(8,7))
sns.barplot(x=subset.value_counts().index, y=subset.value_counts(), color="crimson")
plt.xlabel("PageRank score")
plt.xticks(rotation=45)
plt.ylabel("count")
plt.title(f"PageRank scores distribution: fraudulent nodes", fontsize=16)
plt.show()


# In[159]:


subset = pagerank_scores_classes[pagerank_scores_classes["class"] == 0]["PageRankScore"]
plt.figure(figsize=(8,7))
ax = sns.barplot(x=subset.value_counts().index, y=subset.value_counts(), color="lightseagreen")
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xlabel("PageRank score")
plt.xticks(rotation=45)
plt.ylabel("count")
plt.title(f"PageRank scores distribution: non-fraudulent nodes", fontsize=16)
plt.show()


# ### Betweenness scores distribution

# In[45]:


betweenness_scores = node_metrics.getBetweennessCentrality()
print(betweenness_scores["BetweennessScore"].max())
print(betweenness_scores["BetweennessScore"].min())


# ### Eigenvector scores distribution

# In[46]:


def plotEigenvectorScoresDistribution(subset, color, title_postfix=""):
    subset = subset.value_counts()
    plt.figure(figsize=(8,7))
    sns.barplot(x=subset.index, y=subset, color=color)
    plt.xlabel("eigenvector score")
    plt.xticks(rotation=45)
    plt.ylabel("count")
    plt.title(f"Eigenvector scores distribution{title_postfix}")
    plt.show()


# In[47]:


eigenvector_scores = node_metrics.getEigenvectorCentrality().round(decimals=3)


# In[48]:


eigenvector_scores_classes = pd.merge(eigenvector_scores, nodes_classes, on="id")


# In[49]:


manager.getCentralityMetricsDataFrame("EigenvectorCentrality")


# In[50]:


plotEigenvectorScoresDistribution(eigenvector_scores["EigenvectorCentrality"], color="navy", title_postfix=": all nodes")


# In[51]:


plotEigenvectorScoresDistribution(eigenvector_scores_classes[eigenvector_scores_classes["class"] == 1]["EigenvectorCentrality"], color="navy", title_postfix=": fraudulent nodes")


# In[52]:


plotEigenvectorScoresDistribution(eigenvector_scores_classes[eigenvector_scores_classes["class"] == 1]["EigenvectorCentrality"], color="navy", title_postfix=": non-fraudulent nodes")


# ### Closeness Centrality Distribution

# In[141]:


def plotClosenessScoresDistribution(subset, color, title_postfix=""):
    subset = subset.value_counts()
    plt.figure(figsize=(8,7))
    ax = sns.barplot(x=subset.index, y=subset, color=color)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("closeness score")
    plt.ylabel("count")
    plt.title(f"Closeness scores distribution{title_postfix}", fontsize=16)
    plt.show()


# In[142]:


closeness_scores = node_metrics.getClosenessCentrality()


# In[143]:


closeness_scores_classes = pd.merge(closeness_scores, nodes_classes, on="id")


# In[144]:


manager.getCentralityMetricsDataFrame("ClosenessScore")


# In[145]:


plotClosenessScoresDistribution(closeness_scores["ClosenessScore"], color="orange")


# In[160]:


plotClosenessScoresDistribution(closeness_scores_classes[closeness_scores_classes["class"] == 0]["ClosenessScore"], color="lightseagreen", title_postfix=": non-fraudulent nodes")


# In[147]:


plotClosenessScoresDistribution(closeness_scores_classes[closeness_scores_classes["class"] == 1]["ClosenessScore"], color="crimson", title_postfix=": fraudulent nodes")


# ### HITS scores distribution

# In[60]:


hits_scores = node_metrics.getHITSCentrality()
hits_scores_classes = pd.merge(hits_scores, nodes_classes, on="id")


# In[61]:


hits_scores_classes.round(decimals=3)["authorityScore"].idxmax(), hits_scores_classes.round(decimals=3)["authorityScore"].max()


# In[62]:


hits_scores_classes.iloc[4588]


# In[63]:


hits_scores_classes.round(decimals=3)["hubScore"].idxmax(), hits_scores_classes.round(decimals=3)["hubScore"].max()


# In[64]:


hits_scores_classes.iloc[15]


# In[65]:


manager.getCentralityMetricsDataFrame("authorityScore")


# In[66]:


manager.getCentralityMetricsDataFrame("hubScore")


# In[67]:


hits_scores_classes["authorityScore"].round(decimals=3).value_counts()


# In[68]:


hits_scores_classes[hits_scores_classes["class"] == 1]["authorityScore"].round(decimals=3).value_counts()


# In[69]:


hits_scores_classes[hits_scores_classes["class"] == 0]["authorityScore"].round(decimals=3).value_counts()


# In[70]:


hits_scores_classes["hubScore"].round(decimals=3).value_counts()


# In[71]:


hits_scores_classes[hits_scores_classes["class"] == 1]["hubScore"].round(decimals=3).value_counts()


# In[72]:


hits_scores_classes[hits_scores_classes["class"] == 0]["hubScore"].round(decimals=3).value_counts()


# ## Louvain Communities Analysis

# In[73]:


def getClassesDistributionInAllCommunities(analysis_manager, save=False, output_directory=None):
    communities_distribution = analysis_manager.getNodeClassesInCommunities().groupby("communityId")["class"].value_counts()
    content = pd.DataFrame(communities_distribution)
    content.columns = ["frequency"]
    content = content.reset_index()

    if save & (output_directory is not None):
        try:
            with open(f"{output_directory}/classes_distribution_in_louvain_communities.txt", 'w') as fh:
                fh.write(f"{content.to_string(header=True, index=True)}\n")

        except Exception as e:
            print("Saving the contents failed. Error message: %s" % e)

    return content

def getSuspiciousCommunities(classes_distribution_in_communities):
    """
    :param classes_distribution_in_communities: a dataframe containing all communities' IDs and all transaction classes.
    :return: a dataframe of communities which can be considered suspicious, potentially a fraud ring.
    """
    communities_illicit = classes_distribution_in_communities[classes_distribution_in_communities["class"] == 1]["communityId"]
    communities_illicit = pd.DataFrame(communities_illicit)
    communities_illicit.columns = ["communityId"]
    return communities_illicit

def getClassesDistributionInSuspiciousCommunities(suspicious_communities, classes_distribution_in_communities):
    return pd.merge(suspicious_communities, classes_distribution_in_communities, on="communityId")


# In[74]:


def getSuspiciousCommunitiesFlow(analysis_manager, suspicious_communities):

    transactions_flow = None

    classes_assigned = analysis_manager.getNodeClassesInCommunities()
    node_ids = pd.merge(classes_assigned, suspicious_communities, on="communityId")["id"]
    str1 = f"{list(node_ids)}"

    try:
        gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "paysim"))
        transactions_flow = gds.run_cypher("MATCH (n:node)-[r:TRANSACTION]->(m:node) WHERE n.id IN %s RETURN n.id AS txId1, n.class AS txId1_class, r.cost AS weight, m.id AS txId2, m.class AS txId2_class" % str1)
        gds.close()
    except Exception as e:
        print("Error occurred. Check if the database is online. Error message: %s" % e)

    return transactions_flow


# ### Degree distribution in communities

# In[75]:


communities = manager.detectLouvainCommunities()


# In[76]:


nodes_class = node_metrics.getNodesClasses()


# In[162]:


sum(nodes_class["class"] == 1)


# In[77]:


degree_distribution_in_communities = pd.merge(communities, degree_distribution_df, on="id")
degree_class_in_communities = pd.merge(degree_distribution_in_communities, nodes_class, on="id")
degree_class_in_communities[degree_class_in_communities["class"] == 1]


# In[163]:


degree_class_in_communities


# ### Transactions of each class - frequencies in communities

# In[78]:


communities_classes_distribution = getClassesDistributionInAllCommunities(manager, save=True, output_directory="../")
suspicious_communities = getSuspiciousCommunities(communities_classes_distribution)
getClassesDistributionInSuspiciousCommunities(suspicious_communities, communities_classes_distribution)


# ### Studying the transactions flow

# In[79]:


transaction_flow_in_suspicious_communities = getSuspiciousCommunitiesFlow(manager,
                                                                          suspicious_communities=suspicious_communities)


# In[80]:


transaction_flow_in_suspicious_communities


# As seen below, because the transaction is directed from nameOrig to nameDest, we can see that illicit transactions changed their status, as their counts dropped for txId2 classes. The number of licit transactions (2) increased, the number of unknown class transaction decreased as well.

# In[81]:


def getTransactionsFlowDf(value_counts_transactions):
    transactions_flow_df = pd.DataFrame(value_counts_transactions)
    transactions_flow_df= transactions_flow_df.reset_index()
    transactions_flow_df.columns = ["transaction_class", "frequency"]
    return transactions_flow_df


# In[82]:


flow_from = getTransactionsFlowDf(transaction_flow_in_suspicious_communities.txId1_class.value_counts())
flow_from


# In[83]:


flow_to = getTransactionsFlowDf(transaction_flow_in_suspicious_communities.txId2_class.value_counts())
flow_to


# In[84]:


def plotTransactionsFlowClasses(value_counts_transactions_flow1, value_counts_transactions_flow2):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 6))
    sns.barplot(data=value_counts_transactions_flow1, x="transaction_class", y="frequency", ax=ax[0])
    sns.barplot(data=value_counts_transactions_flow2, x="transaction_class", y="frequency", ax=ax[1])
    ax[0].set_xlabel("Transaction class", fontsize=12)
    ax[1].set_xlabel("Transaction class", fontsize=12)
    ax[0].set_ylabel("Frequency", fontsize=12)
    ax[1].set_ylabel("Frequency", fontsize=12)
    ax[0].set_title("Transaction origin's entity", fontsize=14)
    ax[1].set_title("Transaction destination's entity", fontsize=14)
    plt.tight_layout()
    plt.show()


# In[85]:


plotTransactionsFlowClasses(flow_from, flow_to)


# In[86]:


connector.close()


# ## Mann-Whitney Test

# In[87]:


from scipy.stats import mannwhitneyu


# ### Degree distributions

# In[88]:


fraud_indegree_distribution = degree_distribution_and_class[degree_distribution_and_class["class"] == 1]["in_degree"]
fraud_outdegree_distribution = degree_distribution_and_class[degree_distribution_and_class["class"] == 1]["out_degree"]
non_fraud_indegree_distribution = degree_distribution_and_class[degree_distribution_and_class["class"] == 0]["in_degree"]
non_fraud_outdegree_distribution = degree_distribution_and_class[degree_distribution_and_class["class"] == 0]["out_degree"]


# In[89]:


mw_result = mannwhitneyu(x=fraud_indegree_distribution, y=non_fraud_indegree_distribution, alternative="two-sided")
print("Mann-Whitney U test statistic: %5.3f" % mw_result.statistic)
print("p-value: %5.3f" % mw_result.pvalue)


# In[90]:


fraud_indegree_distribution.value_counts()


# In[91]:


non_fraud_indegree_distribution.value_counts()


# In[92]:


mw_result = mannwhitneyu(x=fraud_outdegree_distribution, y=non_fraud_outdegree_distribution, alternative="two-sided")
print("Mann-Whitney U test statistic: %5.3f" % mw_result.statistic)
print("p-value: %5.3f" % mw_result.pvalue)


# ### PageRank scores distribution

# In[93]:


pagerank_distributions = pd.merge(nodes_classes, pagerank_scores, on="id")


# In[94]:


fraud_pagerank_distribution = pagerank_scores[pagerank_distributions["class"] == 1]["PageRankScore"]
non_fraud_pagerank_distribution = pagerank_distributions[pagerank_distributions["class"] == 0]["PageRankScore"]


# In[95]:


mw_result = mannwhitneyu(x=fraud_pagerank_distribution, y=non_fraud_pagerank_distribution, alternative="two-sided")
print("Mann-Whitney U test statistic: %5.3f" % mw_result.statistic)
print("p-value: %5.3f" % mw_result.pvalue)


# In[96]:


fraud_pagerank_distribution.value_counts().sort_index()


# In[97]:


non_fraud_pagerank_distribution.value_counts().sort_index()


# ## Amounts

# In[98]:


plt.figure(figsize=(8,7))
ax = sns.boxplot(x="isFraud", y="amount", data=edgelist, showfliers=False)
ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
plt.show()


# In[99]:


plt.figure(figsize=(8,7))
ax = sns.boxplot(x="isFraud", y="amount", data=edgelist, showfliers=True, flierprops = dict(markerfacecolor='b', markersize=2))
ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
plt.show()


# ## Notes

# The graph's suspicious communities were visualized using Neo4jBrowser. The query was generated using additional code which prints the list of node IDs from a list and puts that in a formatted string of Cypher query. It is as follows:

# In[100]:


def getCommunitiesNodesList(transaction_flow):
    org = transaction_flow["txId1"]
    dst = transaction_flow["txId2"]

    nodes_in_communities = []
    for node1, node2 in zip(org, dst):
        nodes_in_communities.append(node1)
        nodes_in_communities.append(node2)
    return list(pd.Series(nodes_in_communities).unique())


# In[101]:


nodes_in_communities = getCommunitiesNodesList(getSuspiciousCommunitiesFlow(manager, suspicious_communities=getSuspiciousCommunities(getClassesDistributionInAllCommunities(manager, save=True, output_directory="../"))))
query = f"MATCH (n:node)-[r:TRANSACTION]->(m:node) WHERE n.id IN {nodes_in_communities} OR m.id IN {nodes_in_communities} RETURN n.id AS sourceNodeID, m.id AS destinationNodeID, n.class AS sourceNodeClass, r.isFraud AS transactionCategory, r.type AS transactionType, m.class AS destinationNodeClass"
illicit_nodes_and_transactions = connector.run_cypher(query)
for col in illicit_nodes_and_transactions:
    if (col == "sourceNodeID") | (col == "destinationNodeID") | (col == "transactionType"):
        continue
    else:
        print(illicit_nodes_and_transactions[col].value_counts())


# As seen above, within the communities, 16 out of 29 nodes are known to make fraudulent transactions, 16 out of 29 transactions were illicit, but all destination nodes are known to make illicit transactions.

# In[102]:


illicit_nodes_and_transactions.groupby("transactionCategory")["transactionType"].value_counts()


# In[103]:


node_ids = f"{list(illicit_nodes_and_transactions.destinationNodeID)}"
query = f"MATCH (n:node)-[r:TRANSACTION]->(m:node) WHERE n.id IN {node_ids} RETURN n.id AS sourceID, m.id AS destinationID, n.class AS sourceClass, m.class AS destinationClass, r.isFraud AS transactionCategory"
connector.run_cypher(query)


# As seen above, the trace ends at the destination nodes in communities, however, only some of the transactions were CASH_OUT type
