def problem_p02873(input_data):
    S = eval(input_data)

    lst = [1 if S[0] == "<" else -1]

    for i in range(1, len(S)):

        if S[i] == S[i - 1]:

            lst[-1] += 1 if S[i] == "<" else -1

        else:

            lst.append(-1 if S[i] == ">" else 1)

    ans = [0]

    for i in range(len(lst)):

        if lst[i] > 0:

            ans += list(range(1, lst[i] + 1))

        else:

            ans[-1] = max(ans[-1], -lst[i])

            ans += list(range(-lst[i] - 1, -1, -1))

    return sum(ans)
