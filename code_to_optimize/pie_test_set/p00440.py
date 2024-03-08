def problem_p00440():
    while True:

        try:

            n, k = list(map(int, input().split()))

            if n == 0 and k == 0:
                break

            lst = [0 for _ in range(n)]

            lst2 = []

            flag = 0

            for _ in range(k):

                i = int(eval(input()))

                if not i:

                    flag = 1

                else:

                    lst[i - 1] = 1

            l = -1

            r = -1

            for i in range(n):

                if lst[i] == 0:

                    if r != -1:

                        lst2.append((l, r))

                        l = -1

                        r = -1

                else:

                    if l == -1:

                        l = i

                        r = i

                    else:

                        r += 1

            else:

                if r != -1:

                    lst2.append((l, r))

            ans = 0

            #    print(lst)

            #    print(lst2)

            if not flag:

                for t in lst2:

                    ans = max(ans, t[1] - t[0] + 1)

            else:

                for i in range(len(lst2)):

                    if i == 0:

                        ans = max(ans, lst2[0][1] - lst2[0][0] + 1)

                    elif lst2[i][0] - lst2[i - 1][1] == 2:

                        ans = max(ans, lst2[i][1] - lst2[i - 1][0] + 1)

                    else:

                        ans = max(ans, lst2[i][1] - lst2[i][0] + 1)

            print(ans)

        except EOFError:

            break


problem_p00440()
