def problem_p04021():
    import sys

    input = sys.stdin.readline

    read = sys.stdin.read

    n = int(eval(input()))

    A = list(map(int, read().split()))

    L = sorted([(j, i) for i, j in enumerate(A)])

    count = 0

    for i in range(n):

        if (L[i][1] - i) % 2:

            count += 1

    ans = (count + 1) // 2

    print(ans)


problem_p04021()
