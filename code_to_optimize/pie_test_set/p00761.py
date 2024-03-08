def problem_p00761():
    while True:

        a, l = list(map(int, input().split()))

        if (a | l) == 0:
            break

        dic = {}

        cnt = 0

        dic[a] = cnt

        while True:

            cnt += 1

            Sa = str(a)

            Sa = [x for x in list(Sa) if x is not "0"]

            if len(Sa) is 0:
                Sa.append("0")

            Sa.sort()

            S = int("".join(Sa))

            La = [x for x in Sa]

            La.reverse()

            while len(La) < l:
                La.append("0")

            L = int("".join(La))

            a = L - S

            if a in dic:

                print(dic[a], a, cnt - dic[a])

                break

            else:

                dic[a] = cnt


problem_p00761()
