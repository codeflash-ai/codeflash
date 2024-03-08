def problem_p03565():
    # -*- coding: utf-8 -*-

    """

    Created on Mon Sep 28 02:20:36 2020



    @author: liang

    """

    S = eval(input())

    T = eval(input())

    S = S[::-1]

    T = T[::-1]

    res = list()

    for i in range(len(S) - len(T) + 1):

        flag = True

        for j in range(len(T)):

            if S[i + j] == "?" or S[i + j] == T[j]:

                continue

            else:

                flag = False

        if flag == True:

            ans = ""

            for k in range(len(S)):

                if i <= k <= i + len(T) - 1:

                    # print(T[k-i])

                    ans += T[k - i]

                elif S[k] != "?":

                    # print("B")

                    ans += S[k]

                else:

                    # print("C")

                    ans += "a"

            ans = ans[::-1]

            res.append(ans)

    if res:

        res.sort()

        print((res[0]))

    else:

        print("UNRESTORABLE")

    # print(S)

    # print(T)


problem_p03565()
