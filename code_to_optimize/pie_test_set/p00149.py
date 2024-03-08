def problem_p00149():
    from bisect import bisect

    D = [0.2, 0.6, 1.1]

    L = [0, 0, 0, 0]

    R = [0, 0, 0, 0]

    while True:

        try:

            left, right = list(map(float, input().split()))

        except:

            break

        L[bisect(D, left)] += 1

        R[bisect(D, right)] += 1

    for left, right in zip(L, R)[::-1]:

        print(left, right)


problem_p00149()
