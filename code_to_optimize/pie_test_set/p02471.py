def problem_p02471():
    def gcd(a, b):

        global queue

        r = a % b

        if r:

            d = a // b

            sb = queue.pop()

            sa = queue.pop()

            sc = tuple(map(lambda x, y: x - d * y, sa, sb))

            queue.append(sb)

            queue.append(sc)

            return gcd(b, r)

        else:

            return b

    a, b = list(map(int, input().split()))

    queue = [(1, 0, a), (0, 1, b)]

    g = gcd(a, b)

    x, y, r = queue.pop()

    print((x, y))


problem_p02471()
