def problem_p03293():
    s = eval(input())

    t = eval(input())

    for i in range(len(s)):

        if s[i:] + s[:i] == t:

            print("Yes")

            exit()

    print("No")


problem_p03293()
