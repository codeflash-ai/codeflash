def problem_p03017():
    n, a, b, c, d = list(map(int, input().split()))

    s = eval(input())

    if d > c:

        for i in range(a, d - 1):

            if s[i - 1] == s[i] == "#":

                print("No")

                break

        else:

            print("Yes")

    else:

        for i in range(a, c):

            if (s[i - 1] == "#") & (s[i] == "#"):

                print("No")

                break

        else:

            for j in range(b - 1, d):

                if (s[j - 1] == ".") & (s[j] == ".") & (s[j + 1] == "."):

                    print("Yes")

                    break

            else:

                print("No")


problem_p03017()
