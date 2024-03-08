def problem_p02554():
    N = int(eval(input()))

    ans = 10**N - 2 * 9**N + 8**N

    print((ans % (10**9 + 7)))


problem_p02554()
