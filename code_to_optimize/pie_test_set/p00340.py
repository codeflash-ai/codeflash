def problem_p00340():
    rec = sorted(list(map(int, input().split())))

    print(("yes" if rec[0] == rec[1] and rec[2] == rec[3] else "no"))


problem_p00340()
