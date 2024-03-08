def problem_p03856():
    s = input()[::-1]

    d = ["maerd", "remaerd", "esare", "resare"]

    while s:

        for i in range(4):

            if s.startswith(d[i]):

                s = s[len(d[i]) :]

                break

        else:

            print("NO")

            exit()

    print("YES")


problem_p03856()
