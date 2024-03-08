def problem_p00084():
    import re

    s = re.split("[ .,]", input())

    sl = len(s)

    ans = list()

    for i in range(sl):

        if 2 < len(s[i]) and len(s[i]) < 7:

            ans.append(s[i])

        else:

            pass

    ansl = len(ans)

    for i in range(ansl):

        if i == ansl - 1:

            print(ans[i])

        else:

            print(ans[i], end=(" "))


problem_p00084()
