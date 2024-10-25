from code_to_optimize.final_test_set.topological_sort import Graph


def test_graph_simple():
    g = Graph(6)
    g.addEdge(5, 2)
    g.addEdge(5, 0)
    g.addEdge(4, 0)
    g.addEdge(4, 1)
    g.addEdge(2, 3)
    g.addEdge(3, 1)

    assert g.topologicalSort() == [5, 4, 2, 3, 1, 0]


def test_tree_graph():
    g = Graph(4)
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(0, 3)
    result = g.topologicalSort()
    assert result.index(0) < result.index(1)
    assert result.index(0) < result.index(2)
    assert result.index(0) < result.index(3)


def test_complex_dag():
    g = Graph(6)
    g.addEdge(5, 2)
    g.addEdge(5, 0)
    g.addEdge(4, 0)
    g.addEdge(4, 1)
    g.addEdge(2, 3)
    g.addEdge(3, 1)
    result = g.topologicalSort()
    assert all(result.index(u) < result.index(v) for u, v in [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)])


def test_single_node_graph():
    g = Graph(1)
    assert g.topologicalSort() == [0]
