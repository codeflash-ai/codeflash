MOD = 10**9 + 7

def main():
    while True:
        try:
            h, w = map(int, input().split())
        except EOFError:
            break

        chest = [input() for _ in range(h)]
        dp = [[0 for _ in range(w)] for _ in range(h)]

        dp[0][0] = 1  # Base case

        for i in range(h):
            for j in range(w):
                if chest[i][j] == '.' and (i != 0 or j != 0):
                    dp[i][j] = ((dp[i-1][j] if i > 0 else 0) + (dp[i][j-1] if j > 0 else 0)) % MOD

        print(dp[h-1][w-1])

if __name__ == "__main__":
    main()
