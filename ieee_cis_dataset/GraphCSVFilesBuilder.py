import pandas as pd


from extractAccountSignatures import getMeaningfulAttributes, getUniqueNodes
from prepareGraphData import getData, saveNodesData, saveEdgesData


class GraphCSVFileBuilder:
    def __init__(self, identity_csv, transactions_csv):
        self.identity_csv = identity_csv
        self.transactions_csv = transactions_csv

    def buildCSVFiles(self):
        full_dataset = getMeaningfulAttributes(self.identity_csv, self.transactions_csv)
        unique_nodes = getUniqueNodes(full_dataset=full_dataset)

        data = getData(unique_nodes)

        saveNodesData(data=data)
        saveEdgesData(data=data)


if __name__ == "__main__":
    train_identity = pd.read_csv("data/train_identity.csv")
    train_transaction = pd.read_csv("data/train_transaction.csv")
    gfb = GraphCSVFileBuilder(train_identity, train_transaction)
    gfb.buildCSVFiles()



