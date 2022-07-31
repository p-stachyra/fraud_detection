# Fraudulent financial activity: graph analysis for fraud detection

## Description
This is a repository of the codes used in the master thesis on fraud detection.
It does not contain the data used in the research, as they can be accessed on Kaggle. <br>
[PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) <br>
[IEEE-CIS dataset](https://www.kaggle.com/c/ieee-fraud-detection) <br>
[Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) <br>

## Abstract
This thesis aims to answer the question if graph-based methods can be employed
on available financial datasets with the purpose of detecting illicit financial activities. The data was gathered from three separate data sets – one being a synthetic
PaySim dataset, the second one provided by Vesta in cooperation with the Institute of Electrical and Electronics Engineers (IEEE) and the third one related to
Bitcoin transactions. In all cases, exploratory analysis is applied to attempt to
gain an initial overview of the data sets and presumably to identify certain characteristics which can serve to find additional methods for fraud detection. The data
are analyzed using graph-based approaches which allows for retrieving centrality
metrics for different classes of nodes indicating if they are involved in fraudulent
activity or not. The outcomes were examined using goodness of fit analysis and
descriptive statistics measures to determine if there are differences between groups
of observations. At a general level of metrics distribution in different observation classes, Mann-Whitney U test was employed. Finally, Louvain modularity
was used to gather information regarding dense communities which can constitute
fraud rings. The results of this study suggest that some of the methods presented
in this paper can be useful, however, precise, non-anonymized data must be provided to prove their efficacy. In all our experiments, the centrality metrics did not
perform well for predicting fraud. Without additional information on the entity
making a transaction it is not possible to flag potentially suspicious nodes accurately. <br>

Keywords: financial network, fraud detection, graph properties, centrality, graph
theory.

