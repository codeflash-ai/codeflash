def problem_p02781():
    from functools import lru_cache

    N = int(eval(input()))

    K = int(eval(input()))

    @lru_cache(None)
    def f(N, K):

        if K == 0:
            return 1

        if N < 10:

            if K == 1:
                return N

            return 0

        q, r = divmod(N, 10)

        ret = 0

        ret += f(q, K - 1) * r

        ret += f(q - 1, K - 1) * (9 - r)

        ret += f(q, K)

        return ret

    print((f(N, K)))


problem_p02781()
