def problem_p02640():
    x, y = list(map(int, input().split()))

    flg = False

    for i in range(101):

        for j in range(101):

            if (i * 2) + (j * 4) == y and (i + j) == x:

                print("Yes")

                flg = True

                break

        if flg == True:

            break

    else:

        print("No")


problem_p02640()
