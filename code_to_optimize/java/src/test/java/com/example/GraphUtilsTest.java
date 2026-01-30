package com.example;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class GraphUtilsTest {

    @Test
    void testFindAllPaths() {
        int[][] graph = {
            {0, 1, 1, 0},
            {0, 0, 1, 1},
            {0, 0, 0, 1},
            {0, 0, 0, 0}
        };

        List<List<Integer>> paths = GraphUtils.findAllPaths(graph, 0, 3);
        assertEquals(3, paths.size());
    }

    @Test
    void testHasCycle() {
        int[][] cyclicGraph = {
            {0, 1, 0},
            {0, 0, 1},
            {1, 0, 0}
        };
        assertTrue(GraphUtils.hasCycle(cyclicGraph));

        int[][] acyclicGraph = {
            {0, 1, 0},
            {0, 0, 1},
            {0, 0, 0}
        };
        assertFalse(GraphUtils.hasCycle(acyclicGraph));
    }

    @Test
    void testCountComponents() {
        int[][] graph = {
            {0, 1, 0, 0},
            {1, 0, 0, 0},
            {0, 0, 0, 1},
            {0, 0, 1, 0}
        };
        assertEquals(2, GraphUtils.countComponents(graph));
    }

    @Test
    void testShortestPath() {
        int[][] graph = {
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1},
            {0, 0, 0, 0}
        };
        assertEquals(3, GraphUtils.shortestPath(graph, 0, 3));
        assertEquals(0, GraphUtils.shortestPath(graph, 0, 0));
        assertEquals(-1, GraphUtils.shortestPath(graph, 3, 0));
    }

    @Test
    void testIsBipartite() {
        int[][] bipartite = {
            {0, 1, 0, 1},
            {1, 0, 1, 0},
            {0, 1, 0, 1},
            {1, 0, 1, 0}
        };
        assertTrue(GraphUtils.isBipartite(bipartite));

        int[][] notBipartite = {
            {0, 1, 1},
            {1, 0, 1},
            {1, 1, 0}
        };
        assertFalse(GraphUtils.isBipartite(notBipartite));
    }

    @Test
    void testCalculateInDegrees() {
        int[][] graph = {
            {0, 1, 1},
            {0, 0, 1},
            {0, 0, 0}
        };
        int[] inDegrees = GraphUtils.calculateInDegrees(graph);

        assertEquals(0, inDegrees[0]);
        assertEquals(1, inDegrees[1]);
        assertEquals(2, inDegrees[2]);
    }

    @Test
    void testCalculateOutDegrees() {
        int[][] graph = {
            {0, 1, 1},
            {0, 0, 1},
            {0, 0, 0}
        };
        int[] outDegrees = GraphUtils.calculateOutDegrees(graph);

        assertEquals(2, outDegrees[0]);
        assertEquals(1, outDegrees[1]);
        assertEquals(0, outDegrees[2]);
    }

    @Test
    void testFindReachableNodes() {
        int[][] graph = {
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        };

        List<Integer> reachable = GraphUtils.findReachableNodes(graph, 0);
        assertEquals(3, reachable.size());
        assertTrue(reachable.contains(0));
        assertTrue(reachable.contains(1));
        assertTrue(reachable.contains(2));
    }

    @Test
    void testToEdgeList() {
        int[][] graph = {
            {0, 1, 0},
            {0, 0, 2},
            {3, 0, 0}
        };

        List<int[]> edges = GraphUtils.toEdgeList(graph);
        assertEquals(3, edges.size());
    }
}
