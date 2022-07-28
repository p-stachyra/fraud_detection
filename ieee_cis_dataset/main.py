import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
import time

from extractAccountsCharacteristics import establishConnection, extractAccounts, evaluateDevicesUsage
from getTabularSummaries import getCountsDataFrame

def main():

    start = time.perf_counter()

    # establish the connection with the database
    if len(sys.argv) != 3:
        print("Username and password for Neo4j database required!")
        print("Usage: python main.py username password")

    user = sys.argv[1]
    pwd = sys.argv[2]

    gds = establishConnection(user, pwd)
    print(getFraudAmongMoreDevices(connector=gds))

    all_licit_illicit = evaluateDevicesUsage(gds)

    visualizeIllicitToLicit(all_licit_illicit=all_licit_illicit.iloc[:20])
    visualizeWeightsDistribution(all_licit_illicit=all_licit_illicit, transaction_category="all")
    visualizeWeightsDistribution(all_licit_illicit=all_licit_illicit, transaction_category="licit")
    visualizeWeightsDistribution(all_licit_illicit=all_licit_illicit, transaction_category="illicit")

    # close the connection
    gds.close()

    finish = time.perf_counter()
    time_delta = finish - start
    print("Program execution completed. Total execution time: %5.3fs" % time_delta)


def getAccountsMoreThanOneDevice(connector):
    all_accounts = extractAccounts(connector)
    accounts_with_more_devices = pd.DataFrame(all_accounts[all_accounts["weight"] > 1]["id"])
    merged = pd.merge(accounts_with_more_devices, all_accounts, on="id", how="outer")
    return merged


def getFraudAmongMoreDevices(connector):
    accounts_more_devices = getAccountsMoreThanOneDevice(connector=connector)
    fraud_more_devices = accounts_more_devices[(accounts_more_devices["weight"] == 1) & (accounts_more_devices["isFraud"] == 1)]
    return len(fraud_more_devices) / len(accounts_more_devices)


def visualizeIllicitToLicit(all_licit_illicit):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.barplot(x=all_licit_illicit["weight"],
                y=all_licit_illicit["illicit_to_all"],
                color="lightblue")
    plt.title("Ratio of illicit to licit transactions", fontsize=15)
    plt.xlabel("weight")
    plt.ylabel("ratio")
    plt.tight_layout()
    plt.show()

def visualizeWeightsDistribution(all_licit_illicit, transaction_category):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=all_licit_illicit["weight"],
                    y=all_licit_illicit[transaction_category],
                    color="lightseagreen")
    plt.title(f"Distribution of weights: {transaction_category} transactions",
              fontsize=15)
    plt.xlabel("weight")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

