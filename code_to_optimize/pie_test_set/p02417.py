def problem_p02417():
    import sys

    strs = sys.stdin.readlines()

    for c in range(ord("a"), ord("z") + 1):

        count = 0

        for str in strs:

            count += str.lower().count(chr(c))

        print(("{0} : {1}".format(chr(c), count)))


problem_p02417()
