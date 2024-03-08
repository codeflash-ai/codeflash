def problem_p02409():
    data = [[[0 for r in range(10)] for f in range(3)] for b in range(4)]

    n = int(input())

    for _ in range(n):

        (b, f, r, v) = [int(i) for i in input().split()]

        data[b - 1][f - 1][r - 1] += v

    for b in range(4):

        for f in range(3):

            for r in range(10):

                print("", data[b][f][r], end="")

            print()

        if b != 3:

            print("#" * 20)


problem_p02409()
