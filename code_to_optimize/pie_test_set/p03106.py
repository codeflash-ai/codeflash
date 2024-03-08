def problem_p03106():
    a, b, k = [int(x) for x in input().split()]

    # a,bの最大公約数

    def gcd(A, B):

        if A % B == 0:

            return B

        return gcd(B, A % B)

    # 約数を列挙。

    def make_divisors(n):

        divisors = []

        for i in range(1, int(n**0.5) + 1):

            if n % i == 0:

                divisors.append(i)

                if i != n // i:

                    divisors.append(n // i)

        divisors.sort()

        return divisors

    wk1 = gcd(a, b)

    wk2 = make_divisors(wk1)

    ans = wk2[-k]

    print(ans)


problem_p03106()
