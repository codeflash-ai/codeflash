def problem_p02699(input_data):
    S, W = list(map(int, input_data.split()))

    if S <= W:

        return "unsafe"

    elif S > W:

        return "safe"
