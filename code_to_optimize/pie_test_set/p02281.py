def problem_p02281():
    class Node:

        def __init__(self, node_id, left, right):

            self.node_id = node_id

            self.left = left

            self.right = right

        def pre_order_search(self):

            l = self.left

            r = self.right

            print("", self.node_id, end="")

            if l != -1:

                tree[l].pre_order_search()

            if r != -1:

                tree[r].pre_order_search()

        def in_order_search(self):

            l = self.left

            r = self.right

            if l != -1:

                tree[l].in_order_search()

            print("", self.node_id, end="")

            if r != -1:

                tree[r].in_order_search()

        def post_order_search(self):

            l = self.left

            r = self.right

            if l != -1:

                tree[l].post_order_search()

            if r != -1:

                tree[r].post_order_search()

            print("", self.node_id, end="")

    n = int(input())

    tree = [None for i in range(n)]

    root_set = set(range(n))

    for i in range(n):

        node_id, left, right = map(int, input().split())

        tree[node_id] = Node(node_id, left, right)

        root_set -= set([left, right])

    root = root_set.pop()

    print("Preorder")

    tree[root].pre_order_search()

    print("")

    print("Inorder")

    tree[root].in_order_search()

    print()

    print("Postorder")

    tree[root].post_order_search()

    print()


problem_p02281()
