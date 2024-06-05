def problem_p02754(input_data):
    li = list(map(int, input_data.split()))

    out = int(li[0] / (li[1] + li[2])) * li[1]

    remain = li[0] - int(li[0] / (li[1] + li[2])) * (li[1] + li[2])

    if remain > li[1]:

        out += li[1]

    else:

        out += remain

    return out
