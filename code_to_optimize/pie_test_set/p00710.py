def problem_p00710():
    while True:

        n, r = list(map(int, input().split()))

        if n == 0 and r == 0:

            break

        n_list = []

        for i in range(n):

            n_list.append(i + 1)

        for i in range(r):

            p, c = list(map(int, input().split()))

            n_list = (
                n_list[0 : n - (p - 1) - c]
                + n_list[n - (p - 1) : n]
                + n_list[n - (p - 1) - c : n - (p - 1)]
            )

        print(n_list[n - 1])


problem_p00710()
