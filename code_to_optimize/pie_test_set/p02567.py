def problem_p02567():
    class Segment_Tree:

        def __init__(self, N, calc, unit, L=[]):
            """calcを演算とするN項のSegment Treeを作成



            N:要素数

            calc:演算(2変数関数,モノイド)

            unit:モノイドcalcの単位元 (xe=ex=xを満たすe)

            """

            self.calc = calc

            self.unit = unit

            N = max(N, len(L))

            d = max(1, (N - 1).bit_length())

            k = 2**d

            X = [unit] * (k - 1) + L + [unit] * (k - len(L))

            self.num = k

            self.depth = d

            for i in range(k - 2, -1, -1):

                X[i] = calc(X[2 * i + 1], X[2 * i + 2])

            self.data = X

        def index(self, k, index=0):

            return self.data[(self.num - 1) + (k - index)]

        def update(self, k, x, index=0):
            """第k要素をxに変え,更新を行う.



            k:数列の要素

            x:更新後の値

            """

            m = (self.num - 1) + (k - index)

            self.data[m] = x

            for _ in range(self.depth):

                m = (m - 1) // 2

                self.data[m] = self.calc(self.data[2 * m + 1], self.data[2 * m + 2])

        def sub_array(self, From, To, index=0, left_closed=True, right_closed=True):

            A = From - index + (not left_closed)

            B = To - index - (not right_closed)

            return self.__sub_array_second(A, B + 1, 0, 0, self.num)

        def __sub_array_second(self, a, b, k, l, r):

            if r <= a or b <= l:

                return self.unit

            elif a <= l and r <= b:

                return self.data[k]

            else:

                alpha = self.__sub_array_second(a, b, 2 * k + 1, l, (l + r) // 2)

                beta = self.__sub_array_second(a, b, 2 * k + 2, (l + r) // 2, r)

                return self.calc(alpha, beta)

        def all_prod(self):

            return self.data[0]

        def max_right(self, l, r, cond, index=0):
            """以下の2つをともに満たすxの1つを返す.\n

            (1) r=l or cond(data[l]*data[l+1]*...*d[r-1]):True

            (2) r=x or cond(data[l]*data[l+1]*...*data[r]):False

            ※fが単調減少の時,cond(data[l]*...*data[r-1])を満たす最大のrとなる.



            cond:関数(引数が同じならば結果も同じ)

            cond(unit):True

            0<=l<=r<=n

            """

            l -= index

            assert 0 <= l <= r <= self.num, "添字が範囲外"

            assert cond(self.unit), "単位元が条件を満たさない."

            if l == r:

                return r + index

            l += self.num - 1

            sm = self.unit

            calc = self.calc

            while True:

                while l % 2:

                    l = (l - 1) >> 1

                if not cond(calc(sm, self.data[l])):

                    while l < self.num - 1:

                        l = 2 * l + 1

                        if cond(calc(sm, self.data[l])):

                            sm = calc(sm, self.data[l])

                            l += 1

                    return min(l - (self.num - 1) + index, r)

                sm = calc(sm, self.data[l])

                l += 1

                m = l + 1

                if not (m & (-m) != m):

                    break

            return r + index

    # ================================================

    N, Q = list(map(int, input().split()))

    A = list(map(int, input().split()))

    S = Segment_Tree(N, lambda x, y: max(x, y), -float("inf"), A)

    X = []

    for _ in range(Q):

        T, alpha, beta = list(map(int, input().split()))

        if T == 1:

            S.update(alpha, beta, 1)

        elif T == 2:

            X.append(S.sub_array(alpha, beta, 1))

        else:

            X.append(S.max_right(alpha, N, lambda x: x < beta, 1))

    print(("\n".join(map(str, X))))


problem_p02567()
