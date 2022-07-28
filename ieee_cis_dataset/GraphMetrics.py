from graphdatascience import GraphDataScience
from NodeMetrics import NodeMetrics

class GraphMetrics:
    def __init__(self, gds, graph_name, node_name, relationship_name):
        self.gds = gds
        self.graph_name = graph_name
        self.node_name = node_name
        self.relationship_name = relationship_name

        create_graph = "CALL gds.graph.project ('%s', '%s', '%s')" % (self.graph_name, self.node_name, self.relationship_name)

        check_if_exists = "CALL gds.graph.exists('%s') YIELD graphName, exists" % self.graph_name

        status = gds.run_cypher(check_if_exists)
        if not status["exists"][0]:
            gds.run_cypher(create_graph)

    def getGraphSize(self):
        query_size = "MATCH (n) RETURN COUNT(n) AS Size"

        return self.gds.run_cypher(query_size)["Size"].iloc[0]

    def getNumberOfEdges(self):
        query = "MATCH (n)-[r]->() RETURN COUNT(r)"
        return self.gds.run_cypher(query)

    def getWeaklyConnectedComponents(self):
        query_components = "CALL gds.wcc.stream('%s') YIELD nodeId, componentId" % self.graph_name
        return self.gds.run_cypher(query_components)

    def getStronglyConnectedComponents(self):
        query_components = "CALL gds.alpha.scc.stream('%s') YIELD nodeId, componentId" % self.graph_name
        return self.gds.run_cypher(query_components)

    def getNumberOfWeaklyConnectedComponents(self):
        components_df = self.getWeaklyConnectedComponents()
        return len(components_df)

    def getNumberOfStronglyConnectedComponents(self):
        components_df = self.getStronglyConnectedComponents()
        return len(components_df)

    def getSizesWeaklyConnectedComponents(self):
        components_df = self.getWeaklyConnectedComponents()
        return components_df["componentId"].value_counts()

    def getSizesStronglyConnectedComponents(self):
        components_df = self.getStronglyConnectedComponents()
        return components_df["componentId"].value_counts()

    def getFractionsWeaklyConnectedComponents(self):
        components_df = self.getWeaklyConnectedComponents()
        return components_df["componentId"].value_counts() / len(components_df)

    def getFractionsStronglyConnectedComponents(self):
        components_df = self.getStronglyConnectedComponents()
        return components_df["componentId"].value_counts() / len(components_df)

    def getShortestPaths(self):
        query = "CALL gds.alpha.allShortestPaths.stream('%s') YIELD startNodeId, targetNodeId, distance" % self.graph_name
        return self.gds.run_cypher(query)

    def getCommunities(self):
        query = """
                CALL gds.louvain.stream('%s')
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN gds.util.asNode(nodeId).id AS id, communityId, intermediateCommunityIds
                ORDER BY id ASC
                """ % self.graph_name

        return self.gds.run_cypher(query)

    def getModularity(self):
        query = """
                CALL gds.louvain.mutate('%s', { mutateProperty: 'communityId' })
                YIELD communityCount, modularity, modularities
                """ % self.graph_name

        return self.gds.run_cypher(query)




