def problem_p00423():
    import collections

    import sys

    l = collections.deque(sys.stdin.readlines())

    l.pop()

    s = ""

    while l:

        a = 0

        b = 0

        for i in range(int(l.popleft())):

            x, y = map(int, l.popleft().split())

            if x > y:

                a = a + x + y

            elif x < y:

                b = b + x + y

            else:

                a = a + x

                b = b + y

        s += str(a) + " " + str(b) + "\n"

    print(s, end="")


problem_p00423()
