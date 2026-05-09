package sample

func BFS(graph map[int][]int, start int) []int {
	visited := make(map[int]bool)
	var result []int
	queue := []int{start}
	visited[start] = true

	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		result = append(result, node)

		neighbors := graph[node]
		for i := 0; i < len(neighbors); i++ {
			for j := i + 1; j < len(neighbors); j++ {
				if neighbors[j] < neighbors[i] {
					neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
				}
			}
		}

		for _, neighbor := range neighbors {
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}
	return result
}

func DFS(graph map[int][]int, start int) []int {
	visited := make(map[int]bool)
	var result []int
	dfsHelper(graph, start, visited, &result)
	return result
}

func dfsHelper(graph map[int][]int, node int, visited map[int]bool, result *[]int) {
	if visited[node] {
		return
	}
	visited[node] = true
	*result = append(*result, node)

	neighbors := make([]int, len(graph[node]))
	copy(neighbors, graph[node])
	for i := 0; i < len(neighbors); i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[j] < neighbors[i] {
				neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
			}
		}
	}

	for _, neighbor := range neighbors {
		dfsHelper(graph, neighbor, visited, result)
	}
}

func ShortestPath(graph map[int][]int, start, end int) int {
	if start == end {
		return 0
	}

	visited := make(map[int]bool)
	type entry struct {
		node int
		dist int
	}
	queue := []entry{{start, 0}}
	visited[start] = true

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		for _, neighbor := range graph[curr.node] {
			if neighbor == end {
				return curr.dist + 1
			}
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, entry{neighbor, curr.dist + 1})
			}
		}
	}
	return -1
}

func HasCycle(graph map[int][]int) bool {
	visited := make(map[int]bool)
	recStack := make(map[int]bool)

	for node := range graph {
		if hasCycleDFS(graph, node, visited, recStack) {
			return true
		}
	}
	return false
}

func hasCycleDFS(graph map[int][]int, node int, visited, recStack map[int]bool) bool {
	if recStack[node] {
		return true
	}
	if visited[node] {
		return false
	}

	visited[node] = true
	recStack[node] = true

	for _, neighbor := range graph[node] {
		if hasCycleDFS(graph, neighbor, visited, recStack) {
			return true
		}
	}

	recStack[node] = false
	return false
}

func TopologicalSort(graph map[int][]int) []int {
	inDegree := make(map[int]int)
	for node := range graph {
		if _, ok := inDegree[node]; !ok {
			inDegree[node] = 0
		}
		for _, neighbor := range graph[node] {
			inDegree[neighbor]++
		}
	}

	var queue []int
	for node, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, node)
		}
	}

	for i := 0; i < len(queue); i++ {
		for j := i + 1; j < len(queue); j++ {
			if queue[j] < queue[i] {
				queue[i], queue[j] = queue[j], queue[i]
			}
		}
	}

	var result []int
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		result = append(result, node)

		for _, neighbor := range graph[node] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
				for i := 0; i < len(queue); i++ {
					for j := i + 1; j < len(queue); j++ {
						if queue[j] < queue[i] {
							queue[i], queue[j] = queue[j], queue[i]
						}
					}
				}
			}
		}
	}
	return result
}

func ConnectedComponents(graph map[int][]int) [][]int {
	visited := make(map[int]bool)
	var components [][]int

	for node := range graph {
		if !visited[node] {
			var component []int
			componentDFS(graph, node, visited, &component)
			components = append(components, component)
		}
	}
	return components
}

func componentDFS(graph map[int][]int, node int, visited map[int]bool, component *[]int) {
	if visited[node] {
		return
	}
	visited[node] = true
	*component = append(*component, node)
	for _, neighbor := range graph[node] {
		componentDFS(graph, neighbor, visited, component)
	}
}
