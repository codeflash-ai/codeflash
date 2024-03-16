def problem_p03417(input_data):
    def solve():

        c = list(map(int, input_data.split(" ")))

        if c[0] >= 3 and c[1] >= 3:

            return (c[0] - 2) * (c[1] - 2)

        if c[0] == 1 or c[1] == 1:

            if c[0] == 1 and c[1] == 1:

                return 1

            else:

                return max(max(c) - 2, 0)

        return 0

    return solve()
