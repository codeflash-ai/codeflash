from code_to_optimize.topological_sort import Graph


def test_topological_sort():
    g = Graph(6)
    g.addEdge(5, 2)
    g.addEdge(5, 0)
    g.addEdge(4, 0)
    g.addEdge(4, 1)
    g.addEdge(2, 3)
    g.addEdge(3, 1)

    assert g.topologicalSort() == [5, 4, 2, 3, 1, 0]


def test_topological_sort_2():
    g = Graph(10)

    for i in range(10):
        for j in range(i + 1, 10):
            g.addEdge(i, j)

    assert g.topologicalSort() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    g = Graph(10)

    for i in range(10):
        for j in range(i + 1, 10):
            g.addEdge(i, j)

    assert g.topologicalSort() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_topological_sort_3():
    g = Graph(1000)

    for i in range(1000):
        for j in range(i + 1, 1000):
            g.addEdge(j, i)

    assert g.topologicalSort() == list(reversed(range(1000)))
