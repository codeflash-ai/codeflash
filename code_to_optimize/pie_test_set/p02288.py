def problem_p02288():
    from collections import deque

    class PriorityQueue:

        def __init__(self):

            self.nodes = []

        def max_heapify(self, i):

            if i >= len(self.nodes):
                return

            left, right = (i + 1) * 2 - 1, (i + 1) * 2

            largest = i

            if left < len(self.nodes) and self.nodes[i] < self.nodes[left]:
                largest = left

            if right < len(self.nodes) and self.nodes[largest] < self.nodes[right]:
                largest = right

            if largest != i:

                self.nodes[i], self.nodes[largest] = self.nodes[largest], self.nodes[i]

                self.max_heapify(largest)

        def print_element(self):

            for node in self.nodes:

                print("", node, end="")

            print("")

    n = int(input())

    A = list(map(int, input().split(" ")))

    pq = PriorityQueue()

    pq.nodes = A

    for i in range((n - 1) // 2, -1, -1):

        pq.max_heapify(i)

    pq.print_element()


problem_p02288()
