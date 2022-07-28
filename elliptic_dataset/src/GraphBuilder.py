import os
import pandas as pd


def buildManager(path_edges, path_features, path_classes, time_step=None):

    data_edges = pd.read_csv(path_edges)
    data_features = pd.read_csv(path_features, header=None)
    data_classes = pd.read_csv(path_classes)

    if time_step is None:
        nodes = extractAllTimestepsNodes(data_features=data_features)
        createNodesHeaderFile(nodes_header="id:ID,class:int,label:LABEL", output_directory="../export_all_timesteps")
        createNodesDataFile(data_classes=data_classes, nodes_timestamp_x=nodes,
                            output_directory="../export_all_timesteps")
        edgelist = data_edges
        createEdgesHeaderFile(edges_header=":START_ID,cost:int,:END_ID,:TYPE", output_directory="../export_all_timesteps")
        createEdgesDataFile(edgelist, output_directory="../export_all_timesteps")

    else:
        nodes_timestep = extractTimestepNodes(data_features=data_features, timestep=time_step)

        createNodesHeaderFile(nodes_header="id:ID,class:int,label:LABEL",
                              output_directory=f"../export_timestep_{time_step}")

        createNodesDataFile(data_classes=data_classes,
                            nodes_timestamp_x=nodes_timestep,
                            output_directory=f"../export_timestep_{time_step}")

        edgelist_timestep = getTimestepEdgelist(nodes_timestep_x=nodes_timestep, data_edges=data_edges)
        createEdgesHeaderFile(edges_header=":START_ID,cost:int,:END_ID,:TYPE",
                              output_directory=f"../export_timestep_{time_step}")

        createEdgesDataFile(edgelist_timestep,
                            output_directory=f"../export_timestep_{time_step}")

    return 0


def extractAllTimestepsNodes(data_features):
    nodes_timestep_x = data_features[0]
    nodes_timestep_x = pd.DataFrame(nodes_timestep_x)
    nodes_timestep_x.columns = ["txId"]
    return nodes_timestep_x


def extractTimestepNodes(data_features, timestep=1):
    nodes_timestep_x = data_features[data_features[1] == timestep][0]
    nodes_timestep_x = pd.DataFrame(nodes_timestep_x)
    nodes_timestep_x.columns = ["txId"]
    return nodes_timestep_x


def getTimestepEdgelist(nodes_timestep_x, data_edges):
    # make sure that the transactions exist in a time step
    # select only those records which transaction IDs exist in timestep1 list
    nodes_timestep_x = nodes_timestep_x["txId"].unique()
    txId1_timestep_x = []
    txId2_timestep_x = []
    index = 0
    for i in range(len(data_edges)):
        index += 1
        print(f"Progress: {index} / {len(data_edges)}", end='\r')

        if (data_edges.txId1.iloc[i] in nodes_timestep_x) & (data_edges.txId2.iloc[i] in nodes_timestep_x):
            txId1_timestep_x.append(data_edges.txId1.iloc[i])
            txId2_timestep_x.append(data_edges.txId2.iloc[i])

    data_edges_timestep_x = pd.DataFrame([txId1_timestep_x, txId2_timestep_x]).T
    data_edges_timestep_x.columns = ["txId1", "txId2"]

    return data_edges_timestep_x


def createNodesHeaderFile(nodes_header, output_directory="export"):
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    with open(f"{output_directory}/nodes_header.csv", 'w') as fh:
        fh.write(nodes_header)

    return 0


def createNodesDataFile(data_classes, nodes_timestamp_x, output_directory="export"):

    data_classes_timestep_x = pd.merge(data_classes, nodes_timestamp_x, on="txId")
    data_classes_timestep_x["class"] = data_classes_timestep_x["class"].replace({"unknown" : 3}).astype("uint8")
    data_classes_timestep_x = data_classes_timestep_x.assign(label="node")
    data_classes_timestep_x.to_csv(f"{output_directory}/nodes_data.csv", header=False, index=False)

    return 0


def createEdgesHeaderFile(edges_header, output_directory="export"):
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    with open(f"{output_directory}/edges_header.csv", 'w') as fh:
        fh.write(edges_header)

    return 0


def createEdgesDataFile(edgelist_timestep_x, output_directory="export"):
    # for obtaining relationship properties, the edge properties must be added using the data from the edgelist
    # the properties must be between START ID and END ID
    edges_timestep_x = edgelist_timestep_x[["txId1"]]
    edges_timestep_x = edges_timestep_x.assign(cost=1)
    edges_timestep_x = edges_timestep_x.assign(txId2=edgelist_timestep_x["txId2"])
    edges_timestep_x = edges_timestep_x.assign(Type="TRANSACTION")
    edges_timestep_x.to_csv(f"{output_directory}/edges_data.csv", header=False, index=False)

    return 0
