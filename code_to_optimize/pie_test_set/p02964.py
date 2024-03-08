def problem_p02964():
    def solve():

        N, K = list(map(int, input().split()))

        As = list(map(int, input().split()))

        def getDistNexts(As):

            lenA = len(As)

            poss = dict()

            for i, A in enumerate(As):

                if A not in poss:

                    poss[A] = i + lenA

            distNexts = [0] * (lenA)

            for i in reversed(list(range(lenA))):

                distNexts[i] = poss[As[i]] - i

                poss[As[i]] = i

            return distNexts

        distNexts = getDistNexts(As)

        NK = N * K

        maxD = (NK).bit_length() - 1

        dp = [[0] * (N) for _ in range(maxD + 1)]

        for i in range(N):

            dp[0][i] = distNexts[i] + 1

        for d in range(1, maxD + 1):

            for i in range(N):

                i2 = i + dp[d - 1][i]

                dp[d][i] = i2 + dp[d - 1][i2 % N] - i

        iNow = 0

        dist = 0

        for d in reversed(list(range(maxD + 1))):

            if dist + dp[d][iNow] < NK:

                dist += dp[d][iNow]

                iNow = dist % N

        Ss = []

        setS = set()

        for A in As[iNow:]:

            if A in setS:

                S = ""

                while S != A:

                    S = Ss.pop()

                    setS.remove(S)

            else:

                Ss.append(A)

                setS.add(A)

        if Ss:

            print((" ".join(map(str, Ss))))

    solve()


problem_p02964()
