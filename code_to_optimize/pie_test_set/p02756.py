def problem_p02756():
    S = str(eval(input()))

    Q = int(eval(input()))

    hanten = 0

    front = ""

    end = ""

    for i in range(Q):

        query = list(map(str, input().split()))

        if query[0] == "1":

            hanten += 1

            hanten = hanten % 2

        else:

            if query[1] == "1":

                if hanten == 0:

                    front = query[2] + front

                else:

                    end = end + query[2]

            else:

                if hanten == 0:

                    end = end + query[2]

                else:

                    front = query[2] + front

    ans = front + S + end

    if hanten == 1:

        ans = ans[::-1]

    print(ans)


problem_p02756()
