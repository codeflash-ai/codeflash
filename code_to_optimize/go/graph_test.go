package sample

import (
	"reflect"
	"testing"
)

func TestBFS(t *testing.T) {
	graph := map[int][]int{
		0: {1, 2},
		1: {3},
		2: {3},
		3: {},
	}
	got := BFS(graph, 0)
	want := []int{0, 1, 2, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("BFS = %v, want %v", got, want)
	}
}

func TestBFSSingleNode(t *testing.T) {
	graph := map[int][]int{0: {}}
	got := BFS(graph, 0)
	want := []int{0}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("BFS single = %v, want %v", got, want)
	}
}

func TestDFS(t *testing.T) {
	graph := map[int][]int{
		0: {1, 2},
		1: {3},
		2: {3},
		3: {},
	}
	got := DFS(graph, 0)
	want := []int{0, 1, 3, 2}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("DFS = %v, want %v", got, want)
	}
}

func TestShortestPath(t *testing.T) {
	graph := map[int][]int{
		0: {1, 2},
		1: {3},
		2: {3},
		3: {},
	}

	if got := ShortestPath(graph, 0, 3); got != 2 {
		t.Errorf("ShortestPath(0,3) = %d, want 2", got)
	}
	if got := ShortestPath(graph, 0, 0); got != 0 {
		t.Errorf("ShortestPath(0,0) = %d, want 0", got)
	}
	if got := ShortestPath(graph, 3, 0); got != -1 {
		t.Errorf("ShortestPath(3,0) = %d, want -1", got)
	}
}

func TestHasCycle(t *testing.T) {
	acyclic := map[int][]int{
		0: {1},
		1: {2},
		2: {},
	}
	if HasCycle(acyclic) {
		t.Error("expected no cycle in DAG")
	}

	cyclic := map[int][]int{
		0: {1},
		1: {2},
		2: {0},
	}
	if !HasCycle(cyclic) {
		t.Error("expected cycle")
	}
}

func TestTopologicalSort(t *testing.T) {
	graph := map[int][]int{
		0: {1, 2},
		1: {3},
		2: {3},
		3: {},
	}
	got := TopologicalSort(graph)
	want := []int{0, 1, 2, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TopologicalSort = %v, want %v", got, want)
	}
}

func TestConnectedComponents(t *testing.T) {
	graph := map[int][]int{
		0: {1},
		1: {0},
		2: {3},
		3: {2},
	}
	components := ConnectedComponents(graph)
	if len(components) != 2 {
		t.Errorf("expected 2 components, got %d", len(components))
	}
}
