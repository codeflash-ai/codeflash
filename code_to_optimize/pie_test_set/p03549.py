def problem_p03549():
    n, m = [int(i) for i in input().split()]

    ans = (n - m) * 100

    ans = (ans + 1900 * m) * 2**m

    print(ans)


problem_p03549()
