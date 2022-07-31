import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ExploratoryAnalysis:
    def __init__(self, dataset, transaction_class_attribute, source_node_attribute, destination_node_attribute, timestep_attribute):
        self.dataset = dataset
        self.transaction_class_attribute = transaction_class_attribute
        self.source_node_attribute = source_node_attribute
        self.destination_node_attribute = destination_node_attribute
        self.timestep_attribute = timestep_attribute

        sns.set()

    def getMissingValues(self):
        return self.dataset.isnull().sum()

    def getDatasetOverview(self):
        return self.dataset.head()

    def getNumberOfTransactionClasses(self):
        return self.dataset[self.transaction_class_attribute].nunique()

    def getTransactionClassesValues(self):
        return self.dataset[self.transaction_class_attribute].unique()

    def getTransactionClassesDistribution(self):
        return self.dataset[self.transaction_class_attribute].value_counts()

    def getUniqueNodes(self):
        initial_nodes = list(self.dataset[self.source_node_attribute])

        for node in self.dataset[self.destination_node_attribute]:
            initial_nodes.append(node)

        all_nodes = pd.DataFrame(initial_nodes)[0]

        return all_nodes.unique()

    def getNumberOfEdges(self):
        return len(self.dataset)

    def getNumberOfTimeSteps(self):
        return self.dataset[self.timestep_attribute].nunique()

    def getGraphSizesDistribution(self):
        dist = self.dataset[self.timestep_attribute].value_counts()
        dist = pd.DataFrame(dist).reset_index()
        dist.columns = ["TimeStep", "Size"]
        return dist

    def getFraudDistributionInTimesteps(self):
        fraud_timesteps = self.dataset[self.dataset[self.transaction_class_attribute] == 1][self.timestep_attribute].value_counts()
        fraud_timesteps = fraud_timesteps.reset_index()
        fraud_timesteps.columns = ["TimeStep", "NumberOfFraudulentTransactions"]
        return fraud_timesteps

    def plotFraudInTimesteps(self, save=False, output_dir=None):
        fraud_timesteps = self.getFraudDistributionInTimesteps()
        plt.figure(figsize=(8, 7))
        ax = sns.barplot(data=fraud_timesteps.sort_values("TimeStep"),
                    x="TimeStep",
                    y="NumberOfFraudulentTransactions",
                    color="gold")
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        plt.xlabel("Time step")
        plt.ylabel("Fraudulent transactions")
        plt.title("Fraudulent transaction for each time step", fontsize=15)
        if save & (output_dir is not None):
            try:
                plt.savefig(f"{output_dir}/EDA_fraud_in_timesteps.PNG")
            except Exception as e:
                print("Saving the figure failed. Error message: %s" % e)
        else:
            plt.show()

    def plotSizesOfGraphs(self, save=False, output_dir=None):
        plt.figure(figsize=(8, 7))
        sns.histplot(self.dataset[self.timestep_attribute],  color="lightseagreen")
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
        plt.figure(figsize=(8,7))
        sns.histplot(self.dataset[self.transaction_class_attribute],
                     color="orange")
        plt.title("Transaction classes distribution", fontsize=15)
        plt.xticks([0, 1], ["no fraud (0)", "fraud (1)"])
        plt.ylabel("Count")
        plt.xlabel("Transaction class")
        if save & (output_dir is not None):
            try:
                plt.savefig(f"{output_dir}/EDA_transaction_classes_distribution.PNG")
            except Exception as e:
                print("Saving the figure failed. Error message: %s" % e)
        else:
            plt.show()

    def plotMissingValues(self, missing_values, save=False, output_dir=None):
        df_mv = pd.DataFrame(missing_values).T
        df_mv.index = ["numberOfMissingValues"]
        plt.figure(figsize=(8, 7))
        ax = sns.barplot(data=df_mv)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(rotation=45)
        plt.title(f"Missing values")
        plt.ylabel("Number of missing values")
        plt.xlabel("Attribute name")

        if save & (output_dir is not None):
            try:
                plt.savefig(f"{output_dir}/EDA_missing_values.PNG")
            except Exception as e:
                print("Saving the figure failed. Error message: %s" % e)
        else:
            plt.show()