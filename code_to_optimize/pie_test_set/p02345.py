def problem_p02345():
    import math

    class SegmentTree:

        __slots__ = ["elem_size", "tree_size", "tree", "default", "op"]

        def __init__(self, a: list, default: int, op):

            real_size = len(a)

            self.elem_size = 1 << math.ceil(math.log2(real_size))

            self.tree_size = 2 * self.elem_size

            self.tree = [default] * self.elem_size + a + [default] * (self.elem_size - real_size)

            self.default = default

            self.op = op

            self.init_tree(op)

        def init_tree(self, op) -> None:

            tree = self.tree

            for i in range(self.elem_size - 1, 0, -1):

                left, right = tree[i << 1], tree[(i << 1) + 1]

                # ===== change me =====

                tree[i] = op(left, right)

        def get_value(self, x: int, y: int):

            op = self.op

            l, r = x + self.elem_size, y + self.elem_size

            tree, result = self.tree, self.default

            while l < r:

                if l & 1:

                    # ===== change me =====

                    result = op(tree[l], result)

                    l += 1

                if r & 1:

                    r -= 1

                    # ===== change me =====

                    result = op(tree[r], result)

                l, r = l >> 1, r >> 1

            return result

        def set_value(self, i: int, value: int, op: str = "="):

            tree = self.tree

            k = self.elem_size + i

            if op == "=":

                tree[k] = value

            elif op == "+":

                tree[k] += value

            elif op == "?":

                # ===== change me =====

                tree[k] = value if value < tree[k] else tree[k]

            while k > 1:

                k >>= 1

                left, right = tree[k << 1], tree[(k << 1) + 1]

                # ===== change me =====

                tree[k] = self.op(left, right)

    import sys

    n, q = map(int, input().split())

    tree = SegmentTree([2**31 - 1] * n, 2**31 - 1, min)

    ans = []

    append = ans.append

    for com, x, y in (map(int, l.split()) for l in sys.stdin.readlines()):

        if com == 0:

            tree.set_value(x, y)

        else:

            append(tree.get_value(x, y + 1))

    print(*ans, sep="\n")


problem_p02345()
