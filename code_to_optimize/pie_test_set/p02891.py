def problem_p02891():
    s = eval(input())

    k = int(eval(input()))

    if len(set(s)) == 1:

        print(((len(s) * k) // 2))

        exit()

    ss = s + s

    shoko = 0

    prev = ""

    cnt = 0

    for i in range(len(s)):

        if s[i] == prev:

            cnt += 1

        else:

            shoko += cnt // 2

            cnt = 1

        prev = s[i]

    shoko += cnt // 2

    kosa = 0

    prev = ""

    cnt = 0

    for i in range(len(ss)):

        if ss[i] == prev:

            cnt += 1

        else:

            kosa += cnt // 2

            cnt = 1

        prev = ss[i]

    kosa += cnt // 2

    kosa -= shoko

    print((shoko + (k - 1) * kosa))


problem_p02891()
