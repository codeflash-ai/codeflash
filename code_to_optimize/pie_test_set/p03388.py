def problem_p03388():
    def solve(ab, i):

        return (ab - 1) // i

    q = int(eval(input()))

    for i in range(q):

        a, b = list(map(int, input().split()))

        if a > b:

            a, b = b, a

        ans = a - 1

        start_a = a + 1

        num = int((a * b) ** 0.5)

        while True:

            if solve(a * b, num) == solve(a * b, num + 1):

                break

            else:

                num += 1

        ans = num - 1

        ans += solve(a * b, num) - 1

        if a == b:

            ans += 1

        print(ans)


problem_p03388()
