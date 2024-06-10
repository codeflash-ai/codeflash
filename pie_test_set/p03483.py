def problem_p03483(input_data):
    from collections import Counter

    class BIT:

        def __init__(self, n):

            self.tree = [0] * (n + 1)

            self.size = n

        def sum(self, i):

            s = 0

            while i > 0:

                s += self.tree[i]

                i -= i & -i

            return s

        def add(self, i, x):

            while i <= self.size:

                self.tree[i] += x

                i += i & -i

    S = eval(input_data)

    N = len(S)

    count = Counter(S)

    flag = False

    for v in list(count.values()):

        if v % 2 == 1:

            if flag:

                return -1

                exit()

            else:

                flag = True

    """
    
    文字sのi文字目がどこにあるか、をO(1)で取り出せる辞書が必要
    
    """

    placeAtS = {s: [] for s in list(set(S))}

    for i in range(N):

        s = S[i]

        placeAtS[s].append(i)

    count2 = {s: 0 for s in list(set(S))}

    placeAtT = [None] * (N)

    tmp = 0

    for i in range(N):

        s = S[i]

        count2[s] += 1

        if count2[s] <= count[s] // 2:

            placeAtT[i] = tmp

            mirror = placeAtS[s][count[s] - count2[s]]

            placeAtT[mirror] = N - tmp - 1

            tmp += 1

    for i in range(N):

        if placeAtT[i] == None:

            placeAtT[i] = tmp

    bit = BIT(N)

    ans = 0

    for i in range(N):

        bit.add(placeAtT[i] + 1, 1)

        ans += bit.sum(N) - bit.sum(placeAtT[i] + 1)

    return ans
