import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ExploratoryAnalysis:
    def __init__(self, dataset_edgelist, dataset_transaction_classes, dataset_features):
        self.dataset_edgelist = dataset_edgelist
        self.dataset_transaction_classes = dataset_transaction_classes
        self.dataset_features = dataset_features

        sns.set()

    def getMissingValuesEdgelist(self):
        return self.dataset_edgelist.isnull().sum()

    def getMissingValuesClasses(self):
        return self.dataset_transaction_classes.isnull().sum()

    def getMissingValuesFeatures(self):
        return self.dataset_features.isnull().sum()

    def getEdgelistOverview(self):
        return self.dataset_edgelist.head()

    def getClassesOverview(self):
        return self.dataset_transaction_classes.head()

    def getFeaturesOverview(self):
        return self.dataset_features.head()

    def getNumberOfTransactionClasses(self):
        return self.dataset_transaction_classes[self.dataset_transaction_classes.columns[1]].nunique()

    def getTransactionClassesNames(self):
        return self.dataset_transaction_classes[self.dataset_transaction_classes.columns[1]].unique()

    def getTransactionClassesDistribution(self):
        return self.dataset_transaction_classes[self.dataset_transaction_classes.columns[1]].value_counts()

    def getNumberOfNodes(self):
        return self.dataset_transaction_classes[self.dataset_transaction_classes.columns[0]].nunique()

    def getNumberOfEdges(self):
        return len(self.dataset_edgelist)

    def getNumberOfTimeSteps(self, timestep_attribute):
        return self.dataset_features[timestep_attribute].nunique()

    def getGraphSizesDistribution(self, timestep_attribute):
        dist = self.dataset_features[timestep_attribute].value_counts()
        dist = pd.DataFrame(dist).reset_index()
        dist.columns = ["TimeStep", "Size"]
        return dist

    def getFraudDistributionInTimesteps(self):
        merged = pd.merge(self.dataset_features[[0, 1]], self.dataset_transaction_classes, left_on=0, right_on="txId")
        fraud_timesteps = merged[merged["class"] == "1"][1].value_counts()
        fraud_timesteps = fraud_timesteps.reset_index()
        fraud_timesteps.columns = ["TimeStep", "NumberOfFraudulentTransactions"]
        return fraud_timesteps

    def plotFraudInTimesteps(self, save=False, output_dir=None):
        sns.set_theme(style="whitegrid")
        merged = pd.merge(self.dataset_features[[0, 1]], self.dataset_transaction_classes, left_on=0, right_on="txId")
        fraud_timesteps = merged[merged["class"] == "1"][1].value_counts(normalize=True)
        fraud_timesteps = fraud_timesteps.reset_index()
        fraud_timesteps.columns = ["TimeStep", "NumberOfFraudulentTransactions"]
        plt.figure(figsize=(8, 7))
        ax = sns.barplot(data=fraud_timesteps.sort_values("TimeStep"),
                    x="TimeStep",
                    y="NumberOfFraudulentTransactions",
                    color="gold")
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        plt.xlabel("Time step")
        plt.ylabel("Ratio of fraudulent transactions")
        plt.title("Fraudulent transaction for each time step", fontsize=15)
        if save & (output_dir is not None):
            try:
                plt.savefig(f"{output_dir}/EDA_fraud_in_timesteps.PNG")
            except Exception as e:
                print("Saving the figure failed. Error message: %s" % e)
        else:
            plt.show()

    def plotSizesOfGraphs(self, timestep_attribute, save=False, output_dir=None):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 7))
        sns.histplot(self.dataset_features[timestep_attribute],  color="lightseagreen")
        plt.title("Graph sizes in each time step", fontsize=15)
        plt.ylabel("Number of transactions (nodes)")
        plt.xlabel("Time step")
        if save & (output_dir is not None):
            try:
                plt.savefig(f"{output_dir}/EDA_sizes_of_graphs_in_timesteps.PNG")
            except Exception as e:
                print("Saving the figure failed. Error message: %s" % e)
        else:
            plt.show()

    def plotTransactionClassesDistribution(self, save=False, output_dir=None):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8,7))
        sns.histplot(self.dataset_transaction_classes[self.dataset_transaction_classes.columns[1]],
                     color="orange")
        plt.title("Transaction classes distribution", fontsize=15)
        plt.xticks([0, 1, 2], ["unknown (3)", "licit (2)", "illicit (1)"])
        plt.ylabel("Frequency")
        plt.xlabel("Transaction class")
        if save & (output_dir is not None):
            try:
                plt.savefig(f"{output_dir}/EDA_transaction_classes_distribution.PNG")
            except Exception as e:
                print("Saving the figure failed. Error message: %s" % e)
        else:
            plt.show()

    def plotMissingValues(self, missing_values, df_name, save=False, output_dir=None):
        sns.set_theme(style="whitegrid")
        df_mv = pd.DataFrame(missing_values).T
        df_mv.index = ["numberOfMissingValues"]
        plt.figure(figsize=(8, 7))
        ax = sns.barplot(data=df_mv)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(rotation=45)
        plt.title(f"Missing values: {df_name}")
        plt.ylabel("Number of missing values")
        plt.xlabel("Attribute name")

        if save & (output_dir is not None):
            try:
                plt.savefig(f"{output_dir}/EDA_missing_values_{df_name}.PNG")
            except Exception as e:
                print("Saving the figure failed. Error message: %s" % e)
        else:
            plt.show()