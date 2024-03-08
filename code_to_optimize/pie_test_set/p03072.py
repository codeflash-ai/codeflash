def problem_p03072():
    n = int(eval(input()))

    l = list(map(int, input().split()))

    m = c = 0

    for i in l:

        if m <= i:

            m = i

            c += 1

    print(c)


problem_p03072()
