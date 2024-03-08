def problem_p02814():
    from functools import reduce

    from fractions import gcd

    N, M, *A = list(map(int, open(0).read().split()))

    def lcm(x, y):

        return x * y // gcd(x, y)

    def sub_by_2(n):

        cnt = 0

        while n % 2 == 0:

            n //= 2

            cnt += 1

        return cnt

    lcm_2 = [sub_by_2(n // 2) for n in A]

    if all(lcm_2[0] == v for v in lcm_2):

        num = reduce(lcm, [v // 2 for v in A])

        res = M // num

        print((res // 2 + res % 2))

    else:

        print((0))


problem_p02814()
