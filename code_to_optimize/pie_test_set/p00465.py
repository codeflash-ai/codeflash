def problem_p00465():
    from heapq import heappop as pop

    from heapq import heappush as push

    INF = 1000000000000

    while True:

        R = int(eval(input()))

        if not R:

            break

        w1, h1, x1, y1 = list(map(int, input().split()))

        x1 -= 1

        y1 -= 1

        lst1 = [list(map(int, input().split())) for _ in range(h1)]

        used1 = [[False] * w1 for _ in range(h1)]

        w2, h2, x2, y2 = list(map(int, input().split()))

        x2 -= 1

        y2 -= 1

        lst2 = [list(map(int, input().split())) for _ in range(h2)]

        used2 = [[False] * w2 for _ in range(h2)]

        def bfs(lst, used, que, w, h):

            v, y, x = pop(que)

            if y > 0 and not used[y - 1][x]:

                push(que, (lst[y - 1][x], y - 1, x))

                used[y - 1][x] = True

            if h > y + 1 and not used[y + 1][x]:

                push(que, (lst[y + 1][x], y + 1, x))

                used[y + 1][x] = True

            if x > 0 and not used[y][x - 1]:

                push(que, (lst[y][x - 1], y, x - 1))

                used[y][x - 1] = True

            if w > x + 1 and not used[y][x + 1]:

                push(que, (lst[y][x + 1], y, x + 1))

                used[y][x + 1] = True

            return v

        que = [(1, y1, x1)]

        used1[y1][x1] = True

        rec1 = [[0, 0]]

        Max = 0

        acc = 0

        while que:

            v = bfs(lst1, used1, que, w1, h1)

            acc += 1

            if v > Max:

                rec1.append([v, acc])

                Max = v

            else:

                rec1[-1][1] += 1

        que = [(1, y2, x2)]

        used2[y2][x2] = True

        rec2 = [[0, 0]]

        Max = 0

        acc = 0

        while que:

            v = bfs(lst2, used2, que, w2, h2)

            acc += 1

            if v > Max:

                rec2.append([v, acc])

                Max = v

            else:

                rec2[-1][1] += 1

        end1 = len(rec1)

        end2 = len(rec2)

        ind1 = 0

        ind2 = end2 - 1

        ans = INF

        #  print(rec1)

        #  print(rec2)

        while ind1 < end1 and ind2 > 0:

            r1, sum1 = rec1[ind1]

            r2, sum2 = rec2[ind2]

            if sum1 + sum2 < R:

                ind1 += 1

                continue

            while ind2 > 0 and sum1 + sum2 >= R:

                ind2 -= 1

                r2, sum2 = rec2[ind2]

            if ind2 == 0 and sum1 + sum2 >= R:

                ans = min(ans, r1 + r2)

                break

            else:

                if ind2 < end2 - 1:

                    ind2 += 1

                r2, sum2 = rec2[ind2]

                ans = min(ans, r1 + r2)

            ind1 += 1

        print(ans)


problem_p00465()
