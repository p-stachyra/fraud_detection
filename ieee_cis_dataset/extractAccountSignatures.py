import pandas as pd


def getMeaningfulAttributes(df_train_identity, df_train_transaction):
    meaningful_attributes_transactions = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
    ]

    meaningful_attributes_identity = ["TransactionID", "DeviceType", "DeviceInfo"]

    df_train_identity = df_train_identity[meaningful_attributes_identity]
    df_train_transaction = df_train_transaction[meaningful_attributes_transactions]

    full_dataset = pd.merge(df_train_identity,
                            df_train_transaction,
                            on="TransactionID").dropna(subset=["card1", "card2", "card3", "card4", "card5", "card6"])

    full_dataset.to_csv("fraud_detection_dataset.csv", index=False)

    return full_dataset


def generateSignatures(dataframe, attributes):
    identity_attributes = dataframe[attributes].dropna()
    unique_accounts = identity_attributes.drop_duplicates()

    numeric_part = identity_attributes[["card1", "card2", "card3", "card5"]].astype(int)
    numeric_part = numeric_part.astype(str)
    numeric_part = numeric_part[["card1", "card2", "card3", "card5"]].agg("".join, axis=1)

    first_char = identity_attributes["card4"].str[0]

    second_char = identity_attributes["card6"].apply(len).astype(str)

    result = numeric_part + first_char + second_char

    return result


def getUniqueNodes(full_dataset):
    signatures = pd.DataFrame(generateSignatures(full_dataset, ["card1", "card2", "card3", "card4", "card5", "card6"]))
    signatures.columns = ["AccountName"]
    data_nodes = pd.concat([signatures, full_dataset], axis=1)
    data_nodes.to_csv("node_oriented_dataset.csv", index=False)

    return data_nodes

