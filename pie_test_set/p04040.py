def problem_p04040(input_data):
    from sys import stdin, stdout, stderr, setrecursionlimit

    from functools import lru_cache

    setrecursionlimit(10**7)

    mod = 10**9 + 7

    def solve():

        def binom(n, k):

            res = (modfact[n] * factinv[k]) % mod

            res = (res * factinv[n - k]) % mod

            return res

        h, w, a, b = list(map(int, input_data.split()))

        ans = 0

        modfact = [1] * (h + w)

        factinv = [1] * (h + w)

        for i in range(1, h + w):

            modfact[i] = (i * modfact[i - 1]) % mod

            factinv[i] = (pow(i, mod - 2, mod) * factinv[i - 1]) % mod

        for i in range(h - a):

            ans += (binom(b + i - 1, i) * binom(w + h - b - i - 2, h - i - 1)) % mod

            ans %= mod

        return ans

    """
    
    @lru_cache(maxsize=None)
    
    def binom(n, k):
    
        res = (modfact(n) * factinv(k)) % mod
    
        res = (res * factinv(n - k)) % mod
    
        return res
    
    
    
    @lru_cache(maxsize=None)
    
    def modfact(n):
    
        if n == 0:
    
            return 1
    
    
    
        return (n * modfact(n - 1)) % mod
    
    
    
    @lru_cache(maxsize=None)
    
    def factinv(n):
    
        if n == 0:
    
            return 1
    
    
    
        return (pow(n, mod - 2, mod) * factinv(n - 1)) % mod
    
    """

    if __name__ == "__main__":

        solve()
