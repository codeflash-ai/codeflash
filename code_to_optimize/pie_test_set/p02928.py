def problem_p02928():
    # jsc2019-qualB - Kleene Inversion

    from collections import Counter

    def main():

        N, K = tuple(map(int, input().split()))

        A = tuple(map(int, input().split()))

        MOD = 10**9 + 7

        C = Counter(A)

        x = sum(sum(j > i for j in A) * c for i, c in list(C.items()))

        y = sum(sum(b > a for b in A[i + 1 :]) for i, a in enumerate(A[:-1]))

        ans = (x * K * (1 + K) // 2) % MOD

        ans = (ans - y * K) % MOD

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02928()
