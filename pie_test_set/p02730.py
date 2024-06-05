def problem_p02730(input_data):
    S = eval(input_data)

    N = len(S)

    left = S[: int(N / 2)]

    right = S[int((N + 2) / 2) :]

    S_reversed = "".join(reversed(list(S)))

    left_reversed = "".join(reversed(list(left)))

    right_reversed = "".join(reversed(list(right)))

    if S == S_reversed and left == left_reversed and right == right_reversed:

        return "Yes"

    else:

        return "No"
