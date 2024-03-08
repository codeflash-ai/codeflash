def problem_p03268():
    N, K = list(map(int, input().split()))

    if K % 2 == 1:

        n = N // K

        ans = n**3

    else:

        n1 = N // K

        n2 = 1 + (N - K // 2) // K

        ans = n1**3 + n2**3

    print(ans)


problem_p03268()
