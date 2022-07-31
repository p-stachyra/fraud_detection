import os
import pandas as pd
import sys
import time

from GraphBuilder import buildManager
from AnalysisManager import AnalysisManager

def main():

    start = time.perf_counter()

    if ~(os.path.isdir("../export_all_timesteps")):
        buildManager("../data/elliptic_txs_edgelist.csv",
                     "../data/elliptic_txs_features.csv",
                     "../data/elliptic_txs_classes.csv",
                     time_step=None)



    username = sys.argv[1]
    password = sys.argv[2]

    manager = AnalysisManager(username, password, "../data", outputs_base_location="../")


    communities_distribution = manager.getNodeClassesInCommunities().groupby("communityId")["class"].value_counts()
    content = pd.DataFrame(communities_distribution)
    content.columns = ["frequency"]
    content = content.reset_index()
    communities_illicit = content[content["class"] == 1]["communityId"]
    communities_illicit = pd.DataFrame(communities_illicit)
    communities_illicit.columns = ["communityId"]
    print(pd.merge(communities_illicit, content, on="communityId"))


    if "modularity_score.txt" not in os.listdir("../"):
        with open("../modularity_score.txt", 'w') as fh:
            fh.write(f"{pd.DataFrame(manager.getModularityScore()).to_string(header=True, index=True)}\n")

    finish = time.perf_counter()
    time_delta = finish - start
    print("Program finished. Execution time: %5.3f %s." % (time_delta, "second" if time_delta == 1 else "seconds"))


if __name__ == "__main__":
    main()