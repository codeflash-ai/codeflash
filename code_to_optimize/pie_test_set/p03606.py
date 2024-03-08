def problem_p03606():
    n = int(eval(input()))

    count = 0

    for i in range(n):

        l, r = list(map(int, input().split()))

        count += r - l + 1

    print(count)


problem_p03606()
