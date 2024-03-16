def problem_p03463(input_data):
    from sys import stdin

    n, a, b = [int(x) for x in stdin.readline().rstrip().split()]

    while True:

        if a + 1 != b:

            a += 1

        else:

            return "Borys"

            exit()

        if b - 1 != a:

            b -= 1

        else:

            return "Alice"

            exit()
