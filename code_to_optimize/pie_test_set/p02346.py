def problem_p02346():
    import math

    class SegmentTree:

        __slots__ = ["elem_size", "tree_size", "tree"]

        def __init__(self, a: list, default: int):

            real_size = len(a)

            self.elem_size = 1 << math.ceil(math.log2(real_size))

            self.tree_size = 2 * self.elem_size + 1

            self.tree = (
                [default] * (self.elem_size - 1) + a + [default] * (self.elem_size - real_size)
            )

        def get_range_index(self, x, y, k=0, l_end=0, r_end=None):
            """[x, y], tree[k], [l_end, r_end)"""

            default, tree = 0, self.tree

            if r_end is None:

                r_end = self.elem_size

            if l_end == x and y == r_end - 1:

                return tree[k]

            mid = (l_end + r_end) // 2

            left_y = y if y < mid - 1 else mid - 1

            right_x = x if x > mid else mid

            left = (
                self.get_range_index(x, left_y, 2 * k + 1, l_end, mid) if x <= left_y else default
            )

            right = (
                self.get_range_index(right_x, y, 2 * k + 2, mid, r_end) if right_x <= y else default
            )

            return left + right

        def get_value(self, x, y):

            tree = self.tree

            return self.get_range_index(x, y)

            # return sum(tree[n] for n in index_list)

        def update_tree(self, k: int):

            tree = self.tree

            while k > 0:

                k = (k - 1) // 2

                left, right = tree[2 * k + 1], tree[2 * k + 2]

                tree[k] = left + right

        def set_value(self, i: int, value: int, op: str):

            k = self.elem_size - 1 + i

            if op == "=":

                self.tree[k] = value

            elif op == "+":

                self.tree[k] += value

            self.update_tree(k)

    n, q = list(map(int, input().split()))

    rsq = SegmentTree([0] * n, 0)

    ans = []

    for _ in [0] * q:

        c, x, y = list(map(int, input().split()))

        if c == 0:

            rsq.set_value(x - 1, y, "+")

        else:

            ans.append(rsq.get_value(x - 1, y - 1))

    print(("\n".join([str(n) for n in ans])))


problem_p02346()
