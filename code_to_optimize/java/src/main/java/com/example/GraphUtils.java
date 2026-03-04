package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * Graph algorithms.
 */
public class GraphUtils {

    /**
     * Find all paths between two nodes using DFS.
     *
     * @param graph Adjacency matrix representation
     * @param start Starting node
     * @param end Ending node
     * @return List of all paths (each path is a list of nodes)
     */
    public static List<List<Integer>> findAllPaths(int[][] graph, int start, int end) {
        List<List<Integer>> allPaths = new ArrayList<>();
        if (graph == null || graph.length == 0) {
            return allPaths;
        }

        boolean[] visited = new boolean[graph.length];
        List<Integer> currentPath = new ArrayList<>();
        currentPath.add(start);

        findPathsDFS(graph, start, end, visited, currentPath, allPaths);

        return allPaths;
    }

    private static void findPathsDFS(int[][] graph, int current, int end,
                                     boolean[] visited, List<Integer> currentPath,
                                     List<List<Integer>> allPaths) {
        if (current == end) {
            allPaths.add(new ArrayList<>(currentPath));
            return;
        }

        visited[current] = true;

        for (int next = 0; next < graph.length; next++) {
            if (graph[current][next] != 0 && !visited[next]) {
                currentPath.add(next);
                findPathsDFS(graph, next, end, visited, currentPath, allPaths);
                currentPath.remove(currentPath.size() - 1);
            }
        }

        visited[current] = false;
    }

    /**
     * Check if graph has a cycle using DFS.
     *
     * @param graph Adjacency matrix
     * @return true if graph has a cycle
     */
    public static boolean hasCycle(int[][] graph) {
        if (graph == null || graph.length == 0) {
            return false;
        }

        int n = graph.length;

        for (int start = 0; start < n; start++) {
            boolean[] visited = new boolean[n];
            if (hasCycleDFS(graph, start, -1, visited)) {
                return true;
            }
        }

        return false;
    }

    private static boolean hasCycleDFS(int[][] graph, int node, int parent, boolean[] visited) {
        visited[node] = true;

        for (int neighbor = 0; neighbor < graph.length; neighbor++) {
            if (graph[node][neighbor] != 0) {
                if (!visited[neighbor]) {
                    if (hasCycleDFS(graph, neighbor, node, visited)) {
                        return true;
                    }
                } else if (neighbor != parent) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Count connected components using DFS.
     *
     * @param graph Adjacency matrix
     * @return Number of connected components
     */
    public static int countComponents(int[][] graph) {
        if (graph == null || graph.length == 0) {
            return 0;
        }

        int n = graph.length;
        boolean[] visited = new boolean[n];
        int count = 0;

        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfsVisit(graph, i, visited);
                count++;
            }
        }

        return count;
    }

    private static void dfsVisit(int[][] graph, int node, boolean[] visited) {
        visited[node] = true;

        for (int neighbor = 0; neighbor < graph.length; neighbor++) {
            if (graph[node][neighbor] != 0 && !visited[neighbor]) {
                dfsVisit(graph, neighbor, visited);
            }
        }
    }

    /**
     * Find shortest path using BFS.
     *
     * @param graph Adjacency matrix
     * @param start Starting node
     * @param end Ending node
     * @return Shortest path length, or -1 if no path
     */
    public static int shortestPath(int[][] graph, int start, int end) {
        if (graph == null || graph.length == 0) {
            return -1;
        }

        if (start == end) {
            return 0;
        }

        int n = graph.length;
        boolean[] visited = new boolean[n];
        List<Integer> queue = new ArrayList<>();
        int[] distance = new int[n];

        queue.add(start);
        visited[start] = true;
        distance[start] = 0;

        while (!queue.isEmpty()) {
            int current = queue.remove(0);

            for (int neighbor = 0; neighbor < n; neighbor++) {
                if (graph[current][neighbor] != 0 && !visited[neighbor]) {
                    visited[neighbor] = true;
                    distance[neighbor] = distance[current] + 1;

                    if (neighbor == end) {
                        return distance[neighbor];
                    }

                    queue.add(neighbor);
                }
            }
        }

        return -1;
    }

    /**
     * Check if graph is bipartite using coloring.
     *
     * @param graph Adjacency matrix
     * @return true if bipartite
     */
    public static boolean isBipartite(int[][] graph) {
        if (graph == null || graph.length == 0) {
            return true;
        }

        int n = graph.length;
        int[] colors = new int[n];

        for (int i = 0; i < n; i++) {
            colors[i] = -1;
        }

        for (int start = 0; start < n; start++) {
            if (colors[start] == -1) {
                List<Integer> queue = new ArrayList<>();
                queue.add(start);
                colors[start] = 0;

                while (!queue.isEmpty()) {
                    int node = queue.remove(0);

                    for (int neighbor = 0; neighbor < n; neighbor++) {
                        if (graph[node][neighbor] != 0) {
                            if (colors[neighbor] == -1) {
                                colors[neighbor] = 1 - colors[node];
                                queue.add(neighbor);
                            } else if (colors[neighbor] == colors[node]) {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        return true;
    }

    /**
     * Calculate in-degree of each node.
     *
     * @param graph Adjacency matrix
     * @return Array of in-degrees
     */
    public static int[] calculateInDegrees(int[][] graph) {
        if (graph == null || graph.length == 0) {
            return new int[0];
        }

        int n = graph.length;
        int[] inDegree = new int[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (graph[i][j] != 0) {
                    inDegree[j]++;
                }
            }
        }

        return inDegree;
    }

    /**
     * Calculate out-degree of each node.
     *
     * @param graph Adjacency matrix
     * @return Array of out-degrees
     */
    public static int[] calculateOutDegrees(int[][] graph) {
        if (graph == null || graph.length == 0) {
            return new int[0];
        }

        int n = graph.length;
        int[] outDegree = new int[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (graph[i][j] != 0) {
                    outDegree[i]++;
                }
            }
        }

        return outDegree;
    }

    /**
     * Find all nodes reachable from a given node.
     *
     * @param graph Adjacency matrix
     * @param start Starting node
     * @return List of reachable nodes
     */
    public static List<Integer> findReachableNodes(int[][] graph, int start) {
        List<Integer> reachable = new ArrayList<>();

        if (graph == null || graph.length == 0 || start < 0 || start >= graph.length) {
            return reachable;
        }

        boolean[] visited = new boolean[graph.length];
        dfsCollect(graph, start, visited, reachable);

        return reachable;
    }

    private static void dfsCollect(int[][] graph, int node, boolean[] visited, List<Integer> result) {
        visited[node] = true;
        result.add(node);

        for (int neighbor = 0; neighbor < graph.length; neighbor++) {
            if (graph[node][neighbor] != 0 && !visited[neighbor]) {
                dfsCollect(graph, neighbor, visited, result);
            }
        }
    }

    /**
     * Convert adjacency matrix to edge list.
     *
     * @param graph Adjacency matrix
     * @return List of edges as [from, to, weight]
     */
    public static List<int[]> toEdgeList(int[][] graph) {
        List<int[]> edges = new ArrayList<>();

        if (graph == null || graph.length == 0) {
            return edges;
        }

        for (int i = 0; i < graph.length; i++) {
            for (int j = 0; j < graph[i].length; j++) {
                if (graph[i][j] != 0) {
                    edges.add(new int[]{i, j, graph[i][j]});
                }
            }
        }

        return edges;
    }
}
