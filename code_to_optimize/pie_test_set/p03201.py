def problem_p03201():
    import sys

    F = sys.stdin

    N = int(F.readline().strip("\n"))

    A = list(map(int, F.readline().strip("\n").split()))

    A.sort()

    pair_sum = 0

    A_dict = dict()

    for a in A:

        if a not in A_dict:
            A_dict[a] = 1

        else:
            A_dict[a] += 1

    for i in reversed(list(range(1, N))):

        if A_dict[A[i]] > 0:

            nowA = A[i]

            temp_a = A[i]

            k = 0

            while temp_a > 0:

                temp_a //= 2

                k += 1

            opposite = 2**k - nowA

            if opposite in A_dict:

                if opposite == nowA and A_dict[opposite] > 1:

                    pair_sum += 1

                    A_dict[opposite] -= 2

                elif opposite != nowA and A_dict[opposite] > 0:

                    pair_sum += 1

                    A_dict[opposite] -= 1

                    A_dict[nowA] -= 1

    print(pair_sum)


problem_p03201()
