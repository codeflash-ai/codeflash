def problem_p02606():
    l, r, d = [int(i) for i in input().split()]

    count = 0

    for i in range(l, r + 1):

        if i % d == 0:

            count += 1

    print(count)


problem_p02606()
