def problem_p02575():
    import sys, heapq

    input = sys.stdin.buffer.readline

    def main():

        h, w = list(map(int, input().split()))

        # dp[i] : iの時の最小右移動回数

        dp = [0] * (w + 1)

        # 解候補

        res = [0] * w

        # 解候補から消されるもの

        anti = []

        # Ai = {1:dp[i]はvalid 0:invalid}

        # A1 ... AnのBIT(1-indexed)

        BIT = [0] * (w + 1)

        # A1 ~ Aiまでの和 O(logN)

        def BIT_query(idx):

            res_sum = 0

            while idx > 0:

                res_sum += BIT[idx]

                idx -= idx & (-idx)

            return res_sum

        # Ai += x O(logN)

        def BIT_update(idx, x):

            while idx <= w:

                BIT[idx] += x

                idx += idx & (-idx)

        for i in range(1, w + 1):

            BIT_update(i, 1)

        for i in range(1, h + 1):

            a, b = list(map(int, input().split()))

            # a-1,b+1の値を計算

            x = a - 1

            if x != 0 and x != w + 1 and dp[x] == -1:

                # x以下のvalidな個数

                k = BIT_query(x - 1)

                # k番目のvalidな位置okからXに行く

                if k != 0:

                    ok = x

                    ng = 0

                    while ok - ng > 1:

                        mid = (ok + ng) // 2

                        if BIT_query(mid) == k:

                            ok = mid

                        else:

                            ng = mid

                    # xをvalidにする

                    dp[x] = dp[ok] + (x - ok)

                    BIT_update(x, 1)

                    heapq.heappush(res, dp[x])

            x = b + 1

            if x != 0 and x != w + 1 and dp[x] == -1:

                # x以下のvalidな個数

                k = BIT_query(x - 1)

                # k番目のvalidな位置okからXに行く

                if k != 0:

                    ok = x

                    ng = 0

                    while ok - ng > 1:

                        mid = (ok + ng) // 2

                        if BIT_query(mid) == k:

                            ok = mid

                        else:

                            ng = mid

                    # xをvalidにする

                    dp[x] = dp[ok] + (x - ok)

                    BIT_update(x, 1)

                    heapq.heappush(res, dp[x])

            k = BIT_query(a - 1) + 1

            while True:

                ng = a - 1

                ok = b + 1

                while ok - ng > 1:

                    mid = (ok + ng) // 2

                    if BIT_query(mid) >= k:

                        ok = mid

                    else:

                        ng = mid

                if ok > b or dp[ok] == -1:

                    break

                heapq.heappush(anti, dp[ok])

                dp[ok] = -1

                BIT_update(ok, -1)

            while anti and res and anti[0] == res[0]:

                heapq.heappop(anti)

                heapq.heappop(res)

            if res:

                print((res[0] + i))

            else:

                print((-1))

    if __name__ == "__main__":

        main()


problem_p02575()
