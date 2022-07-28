import matplotlib.pyplot as plt
import sys
import seaborn as sns
from graphdatascience import GraphDataScience

from NodeMetrics import NodeMetrics
from GraphMetrics import GraphMetrics

def main():
    username = sys.argv[1]
    password = sys.argv[2]

    connector = GraphDataScience("bolt://localhost:7687", auth=(username, password))
    node_metrics = NodeMetrics(connector, "elliptic", "node", "TRANSACTION")
    graph_metrics = GraphMetrics(connector, "elliptic", "node", "TRANSACTION")

    sns.histplot(node_metrics.getDegreeDistribution()["degree"], color="orange")
    plt.show()
    print(graph_metrics.getWeaklyConnectedComponents()["componentId"].unique())


if __name__ == "__main__":
    main()
