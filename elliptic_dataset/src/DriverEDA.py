import os
import pandas as pd

from ExploratoryAnalysis import ExploratoryAnalysis


def driver(output_directory, path_edgelist=None, path_classes=None, path_features=None):

    # outputs directory
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # CSV paths
    if (path_edgelist is None):
        path_edgelist = "../data/elliptic_txs_edgelist.csv"
    if (path_classes is None):
        path_classes = "../data/elliptic_txs_classes.csv"
    if (path_features is None):
        path_features = "../data/elliptic_txs_features.csv"

    edgelist = pd.read_csv(path_edgelist)
    classes = pd.read_csv(path_classes)
    features = pd.read_csv(path_features, header=None)
    print(classes.shape)
    print(features.shape)
    print(classes.columns)
    print(features.columns)

    eda = ExploratoryAnalysis(edgelist, classes, features)

    edgelist_head_df = eda.getEdgelistOverview()
    edgelist_head_df = edgelist_head_df.to_string(header=True, index=True)
    classes_head_df = eda.getClassesOverview()
    classes_head_df = classes_head_df.to_string(header=True, index=True)
    features_head_df = eda.getFeaturesOverview()
    features_head_df = features_head_df.to_string(header=True, index=True)

    with open(f"{output_directory}/EDA_report.txt", 'w') as fh:
        fh.write(f"\nFive top records of edgelist dataset\n{edgelist_head_df}\n")
        fh.write(f"\nFive top records of classes dataset\n{classes_head_df}\n")
        fh.write(f"\nFive top records of features dataset\n{features_head_df}\n")
        fh.write(f"\nThe number of nodes: {eda.getNumberOfNodes()}\n")
        fh.write(f"The number of edges: {eda.getNumberOfEdges()}\n")
        fh.write(f"The number of timesteps: {eda.getNumberOfTimeSteps(1)}\n")
        fh.write(f"\nNumber of transaction classes: {eda.getNumberOfTransactionClasses()}\nClasses names:\n")
        for c in eda.getTransactionClassesNames():
            fh.write(f"- {c}\n")
        fh.write(
            f"\nTransaction classes distribution:\n\n{eda.getTransactionClassesDistribution().to_string(header=True, index=True)}\n")
        fh.write(
            f"\nFraudulent transactions in each timestep:\n\n{eda.getFraudDistributionInTimesteps().to_string(header=True, index=True)}\n")
        fh.write(
            f"\nGraph sizes distribution:\n\n{eda.getGraphSizesDistribution(1).to_string(header=True, index=True)}\n")
        fh.write(
            f"\nMissing values in the classes dataset:\n\n{eda.getMissingValuesClasses().to_string(header=True, index=True)}\n")
        fh.write(
            f"\nMissing values in the features dataset:\n\n{eda.getMissingValuesFeatures().to_string(header=True, index=True)}\n")
        fh.write(
            f"\nMissing values in the edgelist dataset:\n\n{eda.getMissingValuesEdgelist().to_string(header=True, index=True)}\n")

    eda.plotTransactionClassesDistribution(save=True, output_dir=output_directory)
    eda.plotFraudInTimesteps(save=True, output_dir=output_directory)
    eda.plotSizesOfGraphs(1, save=True, output_dir=output_directory)
    eda.plotMissingValues(eda.getMissingValuesEdgelist(), "edgelist", save=True, output_dir=output_directory)
    eda.plotMissingValues(eda.getMissingValuesFeatures(), "features", save=True, output_dir=output_directory)
    eda.plotMissingValues(eda.getMissingValuesClasses(), "classes", save=True, output_dir=output_directory)


if __name__ == "__main__":
    driver("../EDA_reports")