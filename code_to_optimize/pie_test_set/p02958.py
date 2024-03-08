def problem_p02958():
    import copy

    def actual(n, P):

        nums_asc = sorted(P)

        if nums_asc == P:

            return "YES"

        N = len(P)

        for i in range(N):

            for j in range(1, N):

                swapped_nums = copy.deepcopy(P)

                swapped_nums[i] = P[j]

                swapped_nums[j] = P[i]

                if swapped_nums == nums_asc:

                    return "YES"

        return "NO"

    n = int(eval(input()))

    P = list(map(int, input().split()))

    print((actual(n, P)))


problem_p02958()
