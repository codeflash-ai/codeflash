def problem_p00450():
    while True:

        n = int(eval(input()))

        if not n:
            break

        count = 0

        flag = -1

        lst = [0]

        for i in range(n):

            a = int(eval(input()))

            if flag == a:

                lst[-1] += 1

            else:

                if i % 2 == 0:

                    lst.append(1)

                    flag = a

                else:

                    if len(lst) == 1:

                        lst[-1] += 1

                        flag = a

                    elif len(lst) >= 2:

                        lst.append(lst.pop() + lst.pop() + 1)

                        flag = a

        ans = 0

        if flag == 0:

            for i in range(-1, len(lst) * (-1) - 1, -2):

                ans += lst[i]

        elif flag == 1:

            for i in range(-2, len(lst) * (-1) - 1, -2):

                ans += lst[i]

        print(ans)


problem_p00450()
