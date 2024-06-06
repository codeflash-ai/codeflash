def problem_p02897(input_data):
    def solve(N):

        odd_count = 0

        for x in range(1, N + 1)[::-1]:

            if x % 2 == 1:

                odd_count += 1

        if N == 1:

            return 1

        return odd_count / N

    N = int(eval(input_data))

    return solve(N)
