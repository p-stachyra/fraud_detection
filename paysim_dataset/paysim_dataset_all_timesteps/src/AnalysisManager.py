import inspect
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from matplotlib.ticker import FormatStrFormatter

from graphdatascience import GraphDataScience
from NodeMetrics import NodeMetrics
from GraphMetrics import GraphMetrics


class AnalysisManager:
    def __init__(self, username, password, data_directory, outputs_base_location="../"):

        sns.set()

        self.data_directory = data_directory
        self.outputs_base_location = outputs_base_location
        self.nodes_characteristics = None
        self.classes_data = None

        if "nodes_characteristics.csv" in os.listdir(self.data_directory):
            self.nodes_characteristics = pd.read_csv(f"{self.data_directory}/nodes_characteristics.csv")

        try:
            self.connector = GraphDataScience("bolt://localhost:7687", auth=(username, password))
        except Exception as ex:
            print("Error establishing connection. Exiting the program.")
            sys.exit(1)

        self.node_metrics = NodeMetrics(self.connector, "elliptic", "node", "TRANSACTION")
        self.graph_metrics = GraphMetrics(self.connector, "elliptic", "node", "TRANSACTION")

    def getNodesMetrics(self, save=True):
        self.classes_data = self.node_metrics.getNodesClasses()

        try:
            degree_distribution = self.node_metrics.getDegreeDistribution()
            merged_classes = pd.merge(self.classes_data, degree_distribution, on="id")
            eigenvector_distribution = self.node_metrics.getEigenvectorCentrality()
            merged_classes = pd.merge(merged_classes, eigenvector_distribution, on="id")
            pagerank_distribution = self.node_metrics.getPageRankScores()
            merged_classes = pd.merge(merged_classes, pagerank_distribution, on="id")
            betweenness_distribution = self.node_metrics.getBetweennessCentrality()
            merged_classes = pd.merge(merged_classes, betweenness_distribution, on="id")
            closeness_distribution = self.node_metrics.getClosenessCentrality()
            merged_classes = pd.merge(merged_classes, closeness_distribution, on="id")
            hits_distribution = self.node_metrics.getHITSCentrality()
            merged_classes = pd.merge(merged_classes, hits_distribution, on="id")

            if save:
                merged_classes.to_csv(f"{self.data_directory}/nodes_characteristics.csv")

            self.nodes_characteristics = merged_classes

        except Exception as ex:
            print("Error.\nException message: %s\nFunction: %s" % (ex, inspect.stack()[0][3]))
            sys.exit(1)

        return merged_classes

    def getCentralityMetricsDataFrame(self, metric_name):

        if not os.path.isdir(f"{self.outputs_base_location}/centrality_metrics_characteristics"):
            os.mkdir(f"{self.outputs_base_location}/centrality_metrics_characteristics")

        # check if the dataframe with centrality metrics already exists
        if self.nodes_characteristics is None:
            # if not, generate one
            self.nodes_characteristics = self.getNodesMetrics()

        # global statistics
        mean_score = self.nodes_characteristics[metric_name].mean()
        median_score = self.nodes_characteristics[metric_name].median()
        std_score = self.nodes_characteristics[metric_name].std()
        min_score = self.nodes_characteristics[metric_name].min()
        max_score = self.nodes_characteristics[metric_name].max()

        header = ["all_classes"]

        for v in self.nodes_characteristics["class"].unique():
            if v == 0:
                header.append("licit")
            elif v == 1:
                header.append("illicit")

        index_names = [f"mean_{metric_name}",
                       f"median_{metric_name}",
                       f"standard_deviation_{metric_name}",
                       f"minimum_value_{metric_name}",
                       f"maximum_value_{metric_name}"]
        means = [mean_score]
        medians = [median_score]
        standard_deviations = [std_score]
        minimum_values = [min_score]
        maximum_values = [max_score]

        for i in range(5):
            for c in self.nodes_characteristics["class"].unique():
                # perfect CASE to use switch (:
                if i == 0:
                    means.append(self.nodes_characteristics[self.nodes_characteristics["class"] == c][{metric_name}].mean()[0])
                elif i == 1:
                    medians.append(self.nodes_characteristics[self.nodes_characteristics["class"] == c][{metric_name}].median()[0])
                elif i == 2:
                    standard_deviations.append(self.nodes_characteristics[self.nodes_characteristics["class"] == c][{metric_name}].std()[0])
                elif i == 3:
                    minimum_values.append(self.nodes_characteristics[self.nodes_characteristics["class"] == c][{metric_name}].min()[0])
                elif i == 4:
                    maximum_values.append(self.nodes_characteristics[self.nodes_characteristics["class"] == c][{metric_name}].max()[0])

        basic_statistics_df = pd.DataFrame([means, medians, standard_deviations, minimum_values, maximum_values],
                                           columns=header,
                                           index=index_names)
        basic_statistics_df.to_csv(f"{self.outputs_base_location}/centrality_metrics_characteristics/{metric_name}_mean_median_std.csv")

        return basic_statistics_df.round(decimals=5)

    def describeCentralityMetricsDistribution(self, metric_name):

        if not os.path.isdir(f"{self.outputs_base_location}/reports"):
            os.mkdir(f"{self.outputs_base_location}/reports")

        basic_statistics = self.getCentralityMetricsDataFrame(metric_name)
        basic_statistics = basic_statistics.round(decimals=5)

        try:
            with open(f"{self.outputs_base_location}/reports/{metric_name}_distribution_entire_graph.txt", 'w') as fh:
                fh.write(f"--- {metric_name} distribution ---\n")
                fh.write(f"Mean {metric_name} distribution: %5.3f\n" % basic_statistics["all_classes"].iloc[0])
                fh.write(f"Median {metric_name} distribution: %5.3f\n" % basic_statistics["all_classes"].iloc[1])
                fh.write(
                    f"Standard deviation of the {metric_name}: %5.3f\n" % basic_statistics["all_classes"].iloc[2])
                fh.write(f"Minimum value of the {metric_name}: %5.3f\n" % basic_statistics["all_classes"].iloc[3])
                fh.write(f"Maximum value of the {metric_name}: %5.3f\n" % basic_statistics["all_classes"].iloc[4])
                fh.write("\n--- table for basic descriptive statistics of the metric ---")
                dfAsString = basic_statistics.to_string(header=True, index=True)
                fh.write(f"\n\n{dfAsString}\n")

        except Exception as ex:
            print("Error.\nException message: %s\nFunction: %s" % (ex, inspect.stack()[0][3]))

        return 0

    def detectLouvainCommunities(self):
        return self.graph_metrics.getCommunities()

    def getNodeClassesInCommunities(self):
        communities = self.detectLouvainCommunities()
        merged = pd.merge(self.node_metrics.getNodesClasses(), communities, on="id")
        return merged

    def getModularityScore(self):
        return self.graph_metrics.getModularity()

    def __del__(self):
        self.connector.close()
        print("Connection closed.")




