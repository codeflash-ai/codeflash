def problem_p03216():

    import sys

    def input():

        return sys.stdin.readline().strip()

    N = int(eval(input()))

    S = eval(input())

    Q = int(eval(input()))

    k = list(map(int, input().split()))

    DM_total = [(0, 0, 0)]

    C_list = []

    for i in range(N):

        if S[i] == "D":

            DM_total.append((DM_total[-1][0] + 1, DM_total[-1][1], DM_total[-1][2]))

        elif S[i] == "M":

            DM_total.append(
                (DM_total[-1][0], DM_total[-1][1] + 1, DM_total[-1][2] + DM_total[-1][0])
            )

        else:

            DM_total.append(DM_total[-1])

        if S[i] == "C":

            C_list.append(i)

    ans_list = [0] * Q

    for c in C_list:

        for q in range(Q):

            if k[q] > c:

                ans_list[q] += DM_total[c + 1][2]

            else:

                ans_list[q] += DM_total[c + 1][2] - DM_total[c + 1 - k[q]][2]

                ans_list[q] -= DM_total[c + 1 - k[q]][0] * (
                    DM_total[c + 1][1] - DM_total[c + 1 - k[q]][1]
                )

    for q in range(Q):

        print((ans_list[q]))


problem_p03216()
