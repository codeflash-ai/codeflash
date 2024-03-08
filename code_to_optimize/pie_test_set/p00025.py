def problem_p00025():
    while True:

        try:

            a = list(map(int, input().split()))

            b = list(map(int, input().split()))

            hit = sum(1 for x, y in zip(a, b) if x == y)

            hits = [x for x, y in zip(a, b) if x == y]

            blow = sum(1 for x in a if x in b and x not in hits)

            print((hit, blow))

        except:

            break


problem_p00025()
