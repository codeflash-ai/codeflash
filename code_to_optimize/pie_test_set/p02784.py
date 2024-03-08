def problem_p02784():
    # -*- coding:utf-8 -*-

    h, n = list(map(int, input().split()))

    for i in map(int, input().split()):

        h -= i

    if h <= 0:

        print("Yes")

    else:

        print("No")


problem_p02784()
