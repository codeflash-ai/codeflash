def problem_p02661():
    from numpy import *

    a, b = median(t := loadtxt(open(0), skiprows=1), 0)

    print((int((a - b) * ~(~len(t) % 2)) + 1))


problem_p02661()
