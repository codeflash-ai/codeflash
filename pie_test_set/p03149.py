def problem_p03149(input_data):
    s = eval(input_data)
    return "YNEOS"[sum(t in s for t in "1479") < 4 :: 2]
