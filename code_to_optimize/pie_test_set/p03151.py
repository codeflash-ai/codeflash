def problem_p03151():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    b = list(map(int, input().split()))

    cnt, nega, posi = 0, 0, []

    for x, y in zip(a, b):

        if x < y:

            nega += y - x

            cnt += 1

        else:

            posi.append(x - y)

    if nega > 0:

        posi.sort(reverse=True)

        for i, x in enumerate(posi):

            nega -= x

            if nega <= 0:

                cnt += i + 1

                break

        else:

            cnt = -1

    print(cnt)


problem_p03151()
