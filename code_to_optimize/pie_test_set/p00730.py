def problem_p00730():
    while 1:

        n, w, d = list(map(int, input().split()))

        if w == 0:
            break

        cake = [[w, d]]

        for loop in range(n):

            p, s = list(map(int, input().split()))

            x, y = cake.pop(p - 1)

            s %= x + y

            if s < x:

                x1, x2 = sorted([s, x - s])

                cake.append([x1, y])

                cake.append([x2, y])

            else:

                s -= x

                y1, y2 = sorted([s, y - s])

                cake.append([x, y1])

                cake.append([x, y2])

        print(" ".join(map(str, sorted([x * y for x, y in cake]))))


problem_p00730()
