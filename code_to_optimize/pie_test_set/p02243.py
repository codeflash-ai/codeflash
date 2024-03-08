def problem_p02243():
    from sys import maxsize

    from heapq import heappop, heappush

    class Node:

        WHITE = 0

        GRAY = 1

        BLACK = 2

        def __init__(self):

            self.adjs = []

            self.color = Node.WHITE

            self.dist = maxsize

        def __lt__(self, other):

            return self.dist < other.dist

        def __repr__(self):

            return "Node({},{},{})".format(self.adjs, self.color, self.dist)

    def read_adj():

        n = int(eval(input()))

        nodes = [Node() for _ in range(n)]

        for i in range(n):

            u, _, *rest = [int(x) for x in input().split()]

            for i in range(0, len(rest) - 1, 2):

                nodes[u].adjs.append((rest[i], rest[i + 1]))

        return nodes

    def shortest_path(nodes):

        h = []

        nodes[0].dist = 0

        nodes[0].color = Node.GRAY

        heappush(h, (0, 0))

        while len(h) > 0:

            min_node, min_dist = heappop(h)

            if nodes[min_node].dist < min_dist:

                continue

            nodes[min_node].color = Node.BLACK

            for adj, c in nodes[min_node].adjs:

                if nodes[min_node].dist + c < nodes[adj].dist:

                    nodes[adj].dist = nodes[min_node].dist + c

                    nodes[adj].color = Node.GRAY

                    heappush(h, (adj, nodes[adj].dist))

    def print_dist(nodes):

        for u, n in enumerate(nodes):

            print((u, n.dist))

    def main():

        nodes = read_adj()

        shortest_path(nodes)

        print_dist(nodes)

    if __name__ == "__main__":

        main()


problem_p02243()
