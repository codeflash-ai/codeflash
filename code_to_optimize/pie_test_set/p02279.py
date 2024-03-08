def problem_p02279():
    """Rooted Trees."""

    class Node:

        def __init__(self, num, parent, *children):

            self.id = num

            self.parent = parent

            self.children = list(children)

            self.depth = 0

        def show_info(self):

            print("node {0}: ".format(self.id), end="")

            print("parent = {0}, ".format(self.parent), end="")

            print("depth = {0}, ".format(self.depth), end="")

            if self.children and self.parent != -1:

                print("internal node, ", end="")

            elif self.parent != -1:

                print("leaf, ", end="")

            else:

                print("root, ", end="")

            print(self.children)

    class RootedTree:

        def __init__(self, n):

            self.nodes = [Node(i, -1) for i in range(n)]

        def set_node(self, i_p):

            i_p = list(map(int, i_p.split()))

            num = i_p[0]

            children = i_p[2:]

            if children:

                self.nodes[num].children = children

                for n in children:

                    self.nodes[n].parent = num

        def set_depth(self):

            for n in self.nodes:

                t_n = n

                while t_n.parent != -1:

                    n.depth += 1

                    t_n = self.nodes[t_n.parent]

        def show_nodes_info(self):

            for n in self.nodes:

                n.show_info()

    import sys

    n = int(sys.stdin.readline())

    T = RootedTree(n)

    for x in sys.stdin.readlines():

        T.set_node(x)

    T.set_depth()

    T.show_nodes_info()


problem_p02279()
