def problem_p03073(input_data):
    S = eval(input_data)

    A = "".join([str(i % 2) for i in range(len(S))])

    B = "".join([str((i + 1) % 2) for i in range(len(S))])

    ans1 = ans2 = 0

    for a, b in zip(S, A):

        if a != b:

            ans1 += 1

    for a, b in zip(S, B):

        if a != b:

            ans2 += 1

    return min(ans1, ans2)
