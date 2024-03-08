def problem_p03272():
    N, i = list(map(int, input().split()))

    count = 0

    while N >= i:

        count += 1

        N -= 1

    print(count)


problem_p03272()
