def problem_p02724():
    x = int(eval(input()))

    # a,b = map(int, input().split())

    # l = list(map(int, input().split()))

    ans = 0

    r5 = int(x / 500)

    x = x - 500 * r5

    r1 = int(x / 5)

    print((r5 * 1000 + r1 * 5))


problem_p02724()
