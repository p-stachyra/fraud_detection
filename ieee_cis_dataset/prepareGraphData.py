import os
import pandas as pd


def getData(node_oriented_df):

    # obtain weights based on usage frequencies
    account_to_device = pd.DataFrame(node_oriented_df.groupby(["AccountName"])["DeviceInfo"].value_counts())

    # assign weight attribute
    account_to_device.columns = ["weight"]
    account_to_device = account_to_device.reset_index()

    # merge weights data and the already available accounts data
    merged = pd.merge(account_to_device,
                      node_oriented_df,
                      on=["AccountName", "DeviceInfo"])

    # save the data frame to the disk
    merged[["AccountName",
            "DeviceInfo",
            "weight",
            "TransactionID",
            "isFraud",
            "TransactionDT",
            "TransactionAmt",
            "ProductCD"]].to_csv("weights_account_to_device.csv")

    return merged


def getNodesData(data):
    accounts = data["AccountName"].unique()
    devices = data["DeviceInfo"].unique()
    nodes = list(accounts)
    labels = []
    for account in accounts:
        labels.append("node")
    for device in devices:
        labels.append("device")
        nodes.append(device)

    header = ["node", "label"]
    nodes_data = pd.DataFrame([nodes, labels]).T
    nodes_data.columns = header

    return nodes_data


def getEdgesData(data):

    edges_data = data[["AccountName",
                       "weight",
                       "TransactionID",
                       "isFraud",
                       "TransactionDT",
                       "TransactionAmt",
                       "ProductCD",
                       "DeviceInfo"]]

    edges_data = edges_data.assign(Type="TRANSACTION")
    return edges_data.drop_duplicates(subset=["AccountName", "DeviceInfo"])


def saveEdgesData(data):

    # create exports directory to copy-paste its contents to Neo4j database folder
    if not os.path.isdir("export"):
        os.mkdir("export")

    # get the data related to edges
    edges_data = getEdgesData(data)

    # save edges data to a CSV file with no header nor index
    edges_data.to_csv("export/edges_data.csv", header=False, index=False)

    # save a CSV header file
    with open("export/edges_header.csv", 'w') as fh:
        fh.write(":START_ID,weight:int,TransactionID:long,isFraud:int,TransactionDT:long,TransactionAmt:float,ProductCD,:END_ID,:TYPE")

    # upon successful exit, return 0
    return 0


def saveNodesData(data):

    # create exports directory to copy-paste its contents to Neo4j database folder
    if not os.path.isdir("export"):
        os.mkdir("export")

    # get the data related to nodes
    nodes_data = getNodesData(data)

    # save edges data to a CSV file with no header nor index
    nodes_data.to_csv("export/nodes_data.csv", header=False, index=False)

    # save a CSV header file
    with open("export/nodes_header.csv", 'w') as fh:
        fh.write("id:ID,label:LABEL")

    # upon successful exit, return 0
    return 0

