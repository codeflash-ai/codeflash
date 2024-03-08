def problem_p02995():
    # -*- coding: utf-8 -*-

    """Snippets for lcm.

    Available functions:

    - lcm: Compute least common multiple of a and b.

    """

    def lcm(a: int, b: int) -> int:
        """Compute least common multiple of a and b.

        Args:

            a: Int of number (greater than 0).

            b: Int of number (greater than 0).

        Returns:

            least common multiple.

        Landau notation: O(log n)

        See:

        https://gist.github.com/endolith/114336/eff2dc13535f139d0d6a2db68597fad2826b53c3

        https://docs.python.org/3/library/sys.html#sys.version_info

        """

        from sys import version_info

        if version_info.major == 3 and version_info.minor >= 5:

            from math import gcd

        else:

            from fractions import gcd

        return a * b // gcd(a, b)

    def main():

        a, b, c, d = list(map(int, input().split()))

        l = lcm(c, d)

        x = b // c - (a - 1) // c

        y = b // d - (a - 1) // d

        z = b // l - (a - 1) // l

        w = b - a + 1

        if c % d == 0 or d % c == 0:

            print((w - x))

        else:

            print((w - (x + y - z)))

    if __name__ == "__main__":

        main()


problem_p02995()
