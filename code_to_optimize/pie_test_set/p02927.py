def problem_p02927(input_data):
    M, D = list(map(int, input_data.split()))

    cnt = 0

    for i in range(22, D + 1):

        d = str(i)

        if int(d[1]) >= 2 and int(d[1]) * int(d[0]) <= M:

            cnt += 1

    return cnt
