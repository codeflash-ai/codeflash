def problem_p03149():
    s = eval(input())
    print(("YNEOS"[sum(t in s for t in "1479") < 4 :: 2]))


problem_p03149()
