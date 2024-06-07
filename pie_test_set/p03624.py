def problem_p03624(input_data):
    s = eval(input_data)

    s = sorted(s)

    ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"

    ans = "None"

    for chr_alpha in ascii_lowercase:

        if chr_alpha not in s:

            ans = chr_alpha

            break

    return ans
