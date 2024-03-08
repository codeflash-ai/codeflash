def problem_p01772():
    S = input()

    if "A" not in S:

        print(-1)

    else:

        ans = "A"

        for s in S[S.index("A") + 1 :]:

            if ans[-1] == "A" and s == "Z":
                ans += s

            if ans[-1] == "Z" and s == "A":
                ans += s

        if ans[-1] == "A":
            ans = ans[:-1]

        print(ans if ans else -1)


problem_p01772()
