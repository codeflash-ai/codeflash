def problem_p02598():
    from math import ceil

    def main():

        N, K = list(map(int, input().split()))

        A = list(map(int, input().split()))

        def is_ok(arg):

            # 条件を満たすかどうか？問題ごとに定義

            B = [int(ceil(a / arg)) - 1 for a in A]

            # print(arg, B)

            if sum(B) <= K:

                return True

            else:

                return False

        def meguru_bisect(ng, ok):
            """

            初期値のng,okを受け取り,is_okを満たす最小(最大)のokを返す

            まずis_okを定義すべし

            ng ok は  とり得る最小の値-1 とり得る最大の値+1

            最大最小が逆の場合はよしなにひっくり返す

            """

            while abs(ok - ng) > 1:

                mid = (ok + ng) // 2

                if is_ok(mid):

                    ok = mid

                else:

                    ng = mid

            return ok

        ans = meguru_bisect(0, int(1e20))

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02598()
