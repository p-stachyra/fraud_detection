import pandas as pd
from graphdatascience import GraphDataScience

from getTabularSummaries import getCountsDataFrame


def establishConnection(username, password):
    gds = GraphDataScience("bolt://localhost:7687", auth=(username, password))
    return gds


def extractAccounts(connector):
    query = """MATCH (n:node)-[r:TRANSACTION]->(m:device) RETURN n.id AS id, r.isFraud AS isFraud, r.weight AS weight, m.id AS deviceId"""
    return connector.run_cypher(query).drop_duplicates(subset="id")


def getAccountsOfCategory(all_accounts, fraud_attribute_name, category):
    return all_accounts[all_accounts[fraud_attribute_name] == category]


def evaluateDevicesUsage(connector):
    all_accounts = extractAccounts(connector=connector)
    suspicious_accounts = getAccountsOfCategory(all_accounts=all_accounts,
                                                fraud_attribute_name="isFraud",
                                                category=1)

    licit_accounts = getAccountsOfCategory(all_accounts=all_accounts,
                                           fraud_attribute_name="isFraud",
                                           category=0)

    all_and_licit = pd.merge(getCountsDataFrame(all_accounts["weight"]),
                             getCountsDataFrame(licit_accounts["weight"]),
                             on="index",
                             how="outer")
    all_and_licit.columns = ["weight", "all", "licit"]

    all_licit_illicit = pd.merge(all_and_licit, getCountsDataFrame(suspicious_accounts["weight"]),
                                 left_on="weight",
                                 right_on="index",
                                 how="outer")
    all_licit_illicit = all_licit_illicit.drop(columns=["index"])
    all_licit_illicit.columns = ["weight", "all", "licit", "illicit"]

    all_licit_illicit["illicit_to_all"] = all_licit_illicit["illicit"].div(all_licit_illicit["all"])
    all_licit_illicit["licit_to_all"] = all_licit_illicit["licit"].div(all_licit_illicit["all"])

    return all_licit_illicit
