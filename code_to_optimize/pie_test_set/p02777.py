def problem_p02777():
    S, T = input().split()

    A, B = list(map(int, input().split()))

    U = eval(input())

    if S == U:

        print((A - 1, B))

    else:

        print((A, B - 1))


problem_p02777()
