def problem_p02951():
    a, b, c = list(map(int, input().split()))

    if a < b + c:

        print((b + c - a))

    else:

        print((0))


problem_p02951()
