def problem_p03101():
    #!/usr/bin/env python3

    import re

    n = eval(input())

    n = re.split(" ", n)

    m = eval(input())

    m = re.split(" ", m)

    a = int(n[0]) * int(n[1])

    b = (int(m[0]) * int(n[1])) + (int(m[1]) * int(n[0]))

    b = a - b

    c = int(m[0]) * int(m[1])

    print((b + c))


problem_p03101()
