def problem_p02914():
    n = int(eval(input()))

    A = list(map(int, input().split()))

    not_important = 0

    for a in A:

        not_important ^= a

    for i in range(n):

        A[i] &= ~not_important

    A.sort()

    rank = 0

    for digit in range(60, -1, -1):

        check_bit = 1 << digit

        for i in range(rank, n):

            if A[i] & check_bit:

                A[rank], A[i] = A[i], A[rank]

                break

        else:

            continue

        for i in range(n):

            if i == rank:
                continue

            if A[i] & check_bit:

                A[i] ^= A[rank]

        rank += 1

    ans = 0

    for i in range(rank):

        ans ^= A[i]

    print((ans * 2 + not_important))


problem_p02914()
