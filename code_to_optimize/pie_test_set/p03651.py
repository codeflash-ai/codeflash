def problem_p03651():
    N, K = list(map(int, input().split()))

    src = list(map(int, input().split()))

    def gcd(a, b):

        a, b = max(a, b), min(a, b)

        if b == 0:

            return a

        return gcd(b, a % b)

    def solve():

        if N == 1:
            return src[0] == K

        mx = max(src)

        if K > mx:
            return False

        g = src[0]

        for i in range(1, N):

            g = gcd(g, src[i])

            if g == 1:

                return True

        return K % g == 0

    print(("POSSIBLE" if solve() else "IMPOSSIBLE"))


problem_p03651()
