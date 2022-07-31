class NodeMetrics:
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

    def getDegreeDistribution(self):
        query_degree = """
                        CALL gds.degree.stream('%s')
                        YIELD nodeId, score AS degree
                        RETURN gds.util.asNode(nodeId).id AS id, degree
                        ORDER BY degree DESC, id ASC
                        """ % self.graph_name

        return self.gds.run_cypher(query_degree)

    def getEigenvectorCentrality(self):
        query_eigenvector = """
                            CALL gds.eigenvector.stream('%s')
                            YIELD nodeId, score AS EigenvectorCentrality
                            RETURN gds.util.asNode(nodeId).id AS id, EigenvectorCentrality
                            ORDER BY EigenvectorCentrality DESC, id ASC
                            """ % self.graph_name

        return self.gds.run_cypher(query_eigenvector)

    def getPageRankScores(self):
        query_pagerank = """
                            CALL gds.pageRank.stream('%s')
                            YIELD nodeId, score
                            RETURN gds.util.asNode(nodeId).id AS id, score AS PageRankScore
                            ORDER BY PageRankScore DESC, id ASC
                            """ % self.graph_name

        return self.gds.run_cypher(query_pagerank)

    def getBetweennessCentrality(self):
        query_betweenness = """
                            CALL gds.betweenness.stream('%s')
                            YIELD nodeId, score
                            RETURN gds.util.asNode(nodeId).id AS id, score AS BetweennessScore
                            ORDER BY BetweennessScore DESC, id ASC
                            """ % self.graph_name

        return self.gds.run_cypher(query_betweenness)

    def getClosenessCentrality(self):
        query_closeness = """
                            CALL gds.beta.closeness.stream('%s')
                            YIELD nodeId, score
                            RETURN gds.util.asNode(nodeId).id AS id, score AS ClosenessScore
                            ORDER BY score DESC""" % self.graph_name

        return self.gds.run_cypher(query_closeness)

    def getNodesClasses(self):
        query = "MATCH (n:node) RETURN n.id AS id, n.class AS class"
        return self.gds.run_cypher(query)

    def getClusteringCoefficient(self):

        check_if_exists = "CALL gds.graph.exists('projected') YIELD graphName, exists"

        graph_projection = """
                            CALL gds.graph.project(
                                'projected', 
                                '%s',
                                {
                                    %s: {
                                        orientation: 'UNDIRECTED'
                                    }
                                }
                            )
                            """ % (self.node_name, self.relationship_name)

        status = self.gds.run_cypher(check_if_exists)
        if not status["exists"][0]:
            self.gds.run_cypher(graph_projection)

        query = """
                CALL gds.localClusteringCoefficient.stream('projected')
                YIELD nodeId, localClusteringCoefficient
                RETURN gds.util.asNode(nodeId).id AS id, localClusteringCoefficient
                ORDER BY localClusteringCoefficient DESC
                """

        return self.gds.run_cypher(query)


