def problem_p03523(input_data):

    import re

    s = eval(input_data)

    pattern = re.compile("(A?KIHA?BA?RA?)")

    ret = pattern.findall(s)

    ret2 = pattern.findall("[^AKIHBR]")

    # return (ret)

    if ret is not None and len(ret) == 1 and len(ret[0]) == len(s):

        ans = "YES"

    else:

        ans = "NO"

    return ans
