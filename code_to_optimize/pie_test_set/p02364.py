def problem_p02364():
    v, e = list(map(int, input().split()))

    adj = [list(map(int, input().split())) for i in range(e)]

    adj.sort(key=lambda x: x[2])

    group = [[i] for i in range(v)]

    key = [i for i in range(v)]

    sum = 0

    for i, j, k in adj:

        if key[i] != key[j]:

            h = key[j]

            group[key[i]] += group[h]

            sum += k

            for t in group[h]:

                key[t] = key[i]

            group[h] = []

            if len(group[key[i]]) == v:

                break

    print(sum)


problem_p02364()
