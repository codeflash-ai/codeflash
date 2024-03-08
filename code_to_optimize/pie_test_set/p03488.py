def problem_p03488():
    # https://ikatakos.com/pot/programming_algorithm/contest_history/atcoder/2017/1216_arc087

    # https://pitsbuffersolution.com/compro/atcoder/arc087d.php

    # 最初の移動以外は、どんな操作手順後であっても、正負いずれの方向にも動けるのがミソ

    def solve(s, x, y):

        X = 0

        Y = 1

        def update(move):

            st = set()

            for p in reachable[st_ind]:

                st.add(p + move)

                st.add(abs(p - move))

            reachable[st_ind] = st

            # 絶対値の等しい正負の値を取りうるので、正だけ扱う

        n = len(s)

        spl = list(map(len, s.split("T")))

        x -= next(spl)

        # ここでxを減らすと、

        # 個々のmove後のx,yは、原点対象になるので、正の座標の到達点だけ扱えばよくなる

        reachable = [{0}, {0}]

        st_ind = 1

        for move in spl:

            update(move)

            st_ind ^= 1

        return (abs(x) in reachable[X]) and (abs(y) in reachable[Y])

        # x,yが負であっても、それに対応した正の値が集合に含まれていれば、到達可能

    if __name__ == "__main__":

        s = eval(input())

        x, y = list(map(int, input().split()))

        print(("Yes" if solve(s, x, y) else "No"))


problem_p03488()
