def problem_p03548():
    X, Y, Z = list(map(int, input().split()))

    for n in range(10**5, 0, -1):

        if X >= n * (Y + Z) + Z:

            break

    print(n)


problem_p03548()
