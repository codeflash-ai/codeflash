def problem_p03633():
    import fractions

    N = int(input())

    T = [int(input()) for _ in range(N)]

    ans = 1

    for i in T:

        ans = ans / fractions.gcd(ans, i) * i

    print(ans)


problem_p03633()
