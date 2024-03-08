def problem_p00064():
    import re

    try:

        ls = []

        result = 0

        while True:

            ls.append(eval(input()))

    except:

        for i in ls:

            for s in re.findall(r"\d+", i):

                result += int(s)

        print(result)


problem_p00064()
