def problem_p03740(input_data):
    return "ABlriocwen"[eval(input_data.replace(" ", "-")) ** 2 < 2 :: 2]
