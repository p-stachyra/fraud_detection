import pandas as pd


def getCountsDataFrame(attribute):
    vc_df = pd.DataFrame(attribute.value_counts())
    vc_df.columns = ["count"]
    vc_df = vc_df.reset_index()
    return vc_df

