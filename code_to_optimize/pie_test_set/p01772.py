def problem_p01772(input_data):
    S = input_data

    if "A" not in S:

        return -1

    else:

        ans = "A"

        for s in S[S.index("A") + 1 :]:

            if ans[-1] == "A" and s == "Z":
                ans += s

            if ans[-1] == "Z" and s == "A":
                ans += s

        if ans[-1] == "A":
            ans = ans[:-1]

        return ans if ans else -1
