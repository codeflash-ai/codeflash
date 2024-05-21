def problem_p03687(input_data):
    s = list(eval(input_data))

    ans = len(s)

    for c in s:

        dist = 0

        dists = []

        for d in s:

            if c != d:

                dist += 1

            else:

                dists.append(dist)

                dist = 0

        ans = min(ans, max(dists + [dist]))

    return ans
