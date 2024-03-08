def problem_p03557():
    import bisect, itertools

    n = int(eval(input()))

    A = sorted(list(map(int, input().split())))

    B = sorted(list(map(int, input().split())))

    C = sorted(list(map(int, input().split())))

    c_num = [0] * n

    for i in range(n):

        c_num[i] = n - bisect.bisect_right(C, B[i])

    c_accum = [0] + list(itertools.accumulate(c_num))

    ans = 0

    for i in range(n):

        b_index = bisect.bisect_right(B, A[i])

        ans += c_accum[-1] - c_accum[b_index]

    print(ans)


problem_p03557()
