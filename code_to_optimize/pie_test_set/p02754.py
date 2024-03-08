def problem_p02754():
    li = list(map(int, input().split()))

    out = int(li[0] / (li[1] + li[2])) * li[1]

    remain = li[0] - int(li[0] / (li[1] + li[2])) * (li[1] + li[2])

    if remain > li[1]:

        out += li[1]

    else:

        out += remain

    print(out)


problem_p02754()
