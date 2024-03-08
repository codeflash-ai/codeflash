def problem_p03625():
    from collections import defaultdict

    n = int(eval(input()))

    a_list = [int(x) for x in input().split()]

    d = defaultdict(int)

    for a in a_list:

        d[a] += 1

    temp_list = sorted([(k, v) for k, v in list(d.items()) if v >= 2], reverse=True)

    if len(temp_list) >= 2:

        print((temp_list[0][0] ** 2 if temp_list[0][1] >= 4 else temp_list[0][0] * temp_list[1][0]))

    else:

        print((0))


problem_p03625()
