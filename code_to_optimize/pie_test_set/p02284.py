def problem_p02284():
    import sys

    class Node:

        __slots__ = ["key", "left", "right"]

        def __init__(self, key):

            self.key = key

            self.left, self.right = None, None

        def __str__(self):

            return " " + str(self.key)

    def insert(z):

        global root

        x, y = root, None

        while x:

            y = x

            if z < x.key:

                x = x.left

            else:

                x = x.right

        if y == None:

            root = Node(z)

        elif z < y.key:

            y.left = Node(z)

        else:

            y.right = Node(z)

    def find(z):

        global root

        x, y = root, None

        while x:

            y = x

            if z < x.key:

                x = x.left

            elif z > x.key:

                x = x.right

            else:

                print("yes")

                return

        print("no")

    def preorder(x):

        return str(x) + preorder(x.left) + preorder(x.right) if x else ""

    def inorder(x):

        return inorder(x.left) + str(x) + inorder(x.right) if x else ""

    eval(input())

    root = None

    for s in sys.stdin:

        if s[0] == "p":

            print((inorder(root)))

            print((preorder(root)))

        elif s[0] == "f":

            find(int(s[5:]))

        else:

            insert(int(s[7:]))


problem_p02284()
