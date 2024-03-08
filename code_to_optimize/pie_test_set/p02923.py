def problem_p02923():
    n = int(eval(input()))

    s = list(map(int, input().split()))

    i = 1

    cnt = 0

    cnt_max = 0

    while i < n:

        if s[i] <= s[i - 1]:

            cnt += 1

        else:

            cnt = 0

        cnt_max = max(cnt, cnt_max)

        i += 1

    print(cnt_max)


problem_p02923()
