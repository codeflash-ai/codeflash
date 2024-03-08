def problem_p00693():
    while True:

        N, M = list(map(int, input().split()))

        if (N, M) == (0, 0):

            break

        rules = []

        for _ in range(N):

            f, p1, p2 = input().split()

            rules += [(f, p1 + p2)]

        rules = rules[::-1]

        def check(s):

            def match(pattern, address):

                for p, a in zip(pattern, address):

                    if p == "?":

                        continue

                    if p != a:

                        return False

                return True

            a1, a2, _ = s.split()

            for flag, pattern in rules:

                if flag == "permit" and match(pattern, a1 + a2):

                    return True

                if flag == "deny" and match(pattern, a1 + a2):

                    return False

        ans = list(filter(check, [input() for _ in range(M)]))

        print(len(ans))

        if len(ans) != 0:

            print("\n".join(ans))


problem_p00693()
