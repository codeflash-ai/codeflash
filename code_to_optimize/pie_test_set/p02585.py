def problem_p02585():
    n, k = list(map(int, input().split()))

    positions = [int(i) for i in input().split()]

    scores = [int(i) for i in input().split()]

    if max(scores) <= 0:

        print((max(scores)))

    else:

        positions = [0] + positions

        scores = [0] + scores

        max_point = -pow(10, 9)

        # print(max_point)

        for i in range(1, n + 1):

            s = []

            f_pos = positions[i]

            score = scores[f_pos]

            s.append(score)

            while f_pos != i:

                f_pos = positions[f_pos]

                score += scores[f_pos]

                s.append(score)

            loop = len(s)

            if loop >= k:

                point = max(s[:k])

            elif s[-1] <= 0:

                point = max(s)

            else:

                loop_c = k // loop

                r = k % loop

                point1 = s[-1] * loop_c

                if r != 0:

                    point1 += max(s[:r])

                point2 = s[-1] * (loop_c - 1)

                point2 += max(0, max(s))

                point = max(point1, point2)

            max_point = max(max_point, point)

        print(max_point)


problem_p02585()
