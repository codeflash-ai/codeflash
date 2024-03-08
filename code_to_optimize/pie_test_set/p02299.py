def problem_p02299():
    def dot(c1, c2):

        return c1.real * c2.real + c1.imag * c2.imag

    def cross(c1, c2):

        return c1.real * c2.imag - c1.imag * c2.real

    def string_to_complex(s):

        x, y = list(map(int, s.split()))

        return x + y * 1j

    def contains(polygon, point):

        flag = False

        for v1, v2 in zip(polygon[0:], polygon[1:]):

            a = v1 - point

            b = v2 - point

            if a.imag > b.imag:

                a, b = b, a

            cross_ab = cross(a, b)

            if a.imag <= 0 and b.imag > 0 and cross_ab > 0:

                flag = not flag

            if cross_ab == 0 and dot(a, b) <= 0:

                return 1

        if flag:

            return 2

        else:

            return 0

    import sys

    file_input = sys.stdin

    n = int(file_input.readline())

    polygon = [string_to_complex(file_input.readline()) for i in range(n)]

    polygon.append(polygon[0])

    q = int(file_input.readline())

    for line in file_input:

        t = string_to_complex(line)

        print((contains(polygon, t)))


problem_p02299()
