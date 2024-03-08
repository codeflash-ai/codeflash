def problem_p02280():
    class Node:

        def __init__(self, num, leftChild, rightChild):

            self.id = num

            self.parent = -1

            self.sibling = -1

            self.degree = 0

            self.depth = 0

            self.height = 0

            self.type = "leaf"

            self.leftChild = leftChild

            self.rightChild = rightChild

        def show_info(self):

            print(
                (
                    "node {}:".format(self.id),
                    "parent = {},".format(self.parent),
                    "sibling = {},".format(self.sibling),
                    "degree = {},".format(self.degree),
                    "depth = {},".format(self.depth),
                    "height = {},".format(self.height),
                    "{}".format(self.type),
                )
            )

    def set_node(i_s):

        num, leftChild, rightChild = list(map(int, i_s.split()))

        node = Node(num, leftChild, rightChild)

        T[num] = node

        T[-1] -= max(0, leftChild) + max(0, rightChild)

    def set_attributes(n_i, parent, sibling, depth):

        node = T[n_i]

        node.parent = parent

        node.sibling = sibling

        node.depth = depth

        if node.leftChild != -1:

            node.degree += 1

            node.type = "internal node"

            set_attributes(node.leftChild, node.id, node.rightChild, depth + 1)

        if node.rightChild != -1:

            node.degree += 1

            node.type = "internal node"

            set_attributes(node.rightChild, node.id, node.leftChild, depth + 1)

        if node.leftChild != -1 and node.rightChild != -1:

            node.height = max(T[node.leftChild].height, T[node.rightChild].height) + 1

        elif node.leftChild != -1:

            node.height = T[node.leftChild].height + 1

        elif node.rightChild != -1:

            node.height = T[node.rightChild].height + 1

    import sys

    n = int(sys.stdin.readline())

    T = [None] * n

    T.append(int(n * (n - 1) / 2))

    for x in sys.stdin.readlines():

        set_node(x)

    set_attributes(T[-1], -1, -1, 0)

    T[T[-1]].type = "root"

    for n in T[:-1]:

        n.show_info()


problem_p02280()
