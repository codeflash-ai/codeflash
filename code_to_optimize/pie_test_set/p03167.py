def problem_p03167():
    from collections import deque

    MOD = 10**9 + 7

    H, W = list(map(int, input().split()))

    G = [eval(input()) for _ in range(H)]

    def main():

        dist = [[-1] * W for _ in range(H)]

        dist[0][0] = 0

        dp = [[0] * W for _ in range(H)]

        for i in range(H):

            if G[i][0] == "#":

                break

            dp[i][0] = 1

        for i in range(W):

            if G[0][i] == "#":

                break

            dp[0][i] = 1

        que = deque([(0, 0)])

        while que:

            h, w = que.popleft()

            for dh, dw in ((1, 0), (0, 1)):

                nh = h + dh

                nw = w + dw

                if nh < 0 or nw < 0 or nh >= H or nw >= W:

                    continue

                if G[nh][nw] == "#":

                    continue

                if dist[nh][nw] != -1:

                    continue

                if nh > 0 and nw > 0:

                    dp[nh][nw] = (dp[nh - 1][nw] + dp[nh][nw - 1]) % MOD

                dist[nh][nw] = dist[h][w] + 1

                que.append((nh, nw))

        print((dp[H - 1][W - 1]))

    if __name__ == "__main__":

        main()


problem_p03167()
