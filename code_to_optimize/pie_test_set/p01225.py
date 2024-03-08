def problem_p01225():
    def rm_123(A):

        if len(A) == 0:
            return 1

        for a in set(A):

            if set([a, a + 1, a + 2]) <= set(A):

                if len(A) == 3:
                    return 1

                A1 = A[:]

                for i in range(3):
                    A1.remove(a + i)

                return judge(A1)

    def rm_111(A):

        if len(A) == 0:
            return 1

        for a in set(A):

            if A.count(a) >= 3:

                if len(A) == 3:
                    return 1

                A1 = A[:]

                for i in range(3):
                    A1.remove(a)

                return judge(A1)

    def judge(A):

        return rm_123(A) or rm_111(A)

    for _ in range(eval(input())):

        n = list(map(int, input().split()))

        R = []
        G = []
        B = []

        s = input().split()

        for i in range(len(n)):

            if s[i] == "R":
                R.append(n[i])

            elif s[i] == "G":
                G.append(n[i])

            elif s[i] == "B":
                B.append(n[i])

        print(1 if judge(R) and judge(G) and judge(B) else 0)


problem_p01225()
