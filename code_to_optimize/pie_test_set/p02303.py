def problem_p02303():
    #!/usr/bin/env python

    # -*- coding: utf-8 -*-

    """

    input:

    3

    0.0 0.0

    2.0 0.0

    1.0 1.0



    output:

    1.41421356237

    """

    import math

    import sys

    from operator import attrgetter

    class ClosestPair(object):

        def __init__(
            self,
        ):
            """

            Init closest pairs points set.

            """

            _input = sys.stdin.readlines()

            p_num = int(_input[0])

            points = [x.split() for x in _input[1:]]

            p_list = [complex(float(x), float(y)) for x, y in points]

            p_list.sort(key=attrgetter("real"))

            # assert len(p_list) == p_num

            # print(p_list)

            ans = self.closest_pair(array=p_list, array_length=p_num)

            print(("{:.6f}".format(ans)))

        def closest_pair(self, array, array_length):

            if array_length <= 1:

                return float("inf")

            mid = array_length // 2

            mid_real = array[mid].real

            d = min(
                self.closest_pair(array[:mid], mid).real,
                self.closest_pair(array[mid:], array_length - mid).real,
            )

            return self.brute_force(array, mid_real, d)

        @staticmethod
        def brute_force(array, mid_real, d=float("inf")):

            array.sort(key=attrgetter("imag"))

            min_stack = list()

            for ele in array:

                size = len(min_stack)

                if abs(ele.real - mid_real) >= d:

                    continue

                for j in range(size):

                    dx = ele.real - min_stack[size - j - 1].real

                    dy = ele.imag - min_stack[size - j - 1].imag

                    if dy >= d:

                        break

                    d = min(d, math.sqrt(dx**2 + dy**2))

                min_stack.append(ele)

            return d

    if __name__ == "__main__":

        case = ClosestPair()


problem_p02303()
