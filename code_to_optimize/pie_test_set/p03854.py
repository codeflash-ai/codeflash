def problem_p03854(input_data):
    s = eval(input_data)

    t = ""

    while len(t) < len(s):

        target_len = len(s) - len(t)

        if s[len(t)] == "d":

            if target_len == 5:

                t += "dream"

            elif target_len == 7:

                t += "dreamer"

            elif target_len > 7:

                if s[len(t) + 5] != "d" and s[len(t) + 5 : len(t) + 7 + 1] != "era":

                    t += "dreamer"

                else:

                    t += "dream"

            else:

                break

        else:

            if target_len == 5:

                t += "erase"

            elif target_len == 6:

                t += "eraser"

            elif target_len > 6:

                if s[len(t) + 5] != "d" and s[len(t) + 5 : len(t) + 6 + 2] != "era":

                    t += "eraser"

                else:

                    t += "erase"

            else:

                break

        if s[0 : len(t)] != t:

            break

    return "YES" if s == t else "NO"
