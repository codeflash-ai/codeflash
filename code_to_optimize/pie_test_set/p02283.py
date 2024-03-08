def problem_p02283():
    import sys

    class BinaryTree:

        def __init__(self):

            self.root = None

        def b_insert(self, x):

            if self.root is None:

                self.root = Node(x)

                return

            node = None

            root = self.root

            while root is not None:

                node = root

                if x == root.data:

                    return

                elif x < root.data:

                    root = root.left

                else:

                    root = root.right

            if node is None:

                # print("center",x)

                node = Node(x)

            elif node.data < x:

                # print("right",x)

                node.right = Node(x)

            else:

                # print("left",x)

                node.left = Node(x)

        def b_inorder(self):

            return inorder(self.root)

        def b_preorder(self):

            return preorder(self.root)

    class Node:

        def __init__(self, x):

            self.data = x

            self.left = None

            self.right = None

    def inorder(node):

        nodelist = []

        if node.left:

            nodelist += inorder(node.left)

        nodelist += [node.data]

        if node.right:

            nodelist += inorder(node.right)

        return nodelist

    def preorder(node):

        nodelist = [node.data]

        if node.left:

            nodelist += preorder(node.left)

        if node.right:

            nodelist += preorder(node.right)

        return nodelist

    line = sys.stdin.readline()

    bfs = BinaryTree()

    for temp in sys.stdin:

        if temp[0] == "i":

            command, num = (n for n in temp.split(" "))

            bfs.b_insert(int(num))

        else:

            print("", end=" ")

            print(*bfs.b_inorder())

            print("", end=" ")

            print(*bfs.b_preorder())


problem_p02283()
