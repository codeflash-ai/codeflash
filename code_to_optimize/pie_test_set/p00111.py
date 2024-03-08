def problem_p00111():
    #!/usr/bin/env python

    from sys import stdin, exit

    code_table = {chr(65 + i): "{:05b}".format(i) for i in range(26)}

    code_table.update(
        {" ": "11010", ".": "11011", ",": "11100", "-": "11101", "'": "11110", "?": "11111"}
    )

    def decrypt(it):

        if int(next(it)):

            if int(next(it)):

                if int(next(it)):

                    return "P"

                else:

                    return "E"

            else:

                if int(next(it)):

                    return " "

                else:

                    if int(next(it)):

                        if int(next(it)):

                            if int(next(it)):

                                if int(next(it)):

                                    if int(next(it)):

                                        return "Q"

                                    else:

                                        return "N"

                                else:

                                    if int(next(it)):

                                        return "V"

                                    else:

                                        return "U"

                            else:

                                if int(next(it)):

                                    if int(next(it)):

                                        return "G"

                                    else:

                                        return "B"

                                else:

                                    if int(next(it)):

                                        return "M"

                                    else:

                                        return "J"

                        else:

                            if int(next(it)):

                                return "A"

                            else:

                                if int(next(it)):

                                    if int(next(it)):

                                        return "Y"

                                    else:

                                        return "X"

                                else:

                                    if int(next(it)):

                                        return "-"

                                    else:

                                        return "Z"

                    else:

                        return "R"

        else:

            if int(next(it)):

                if int(next(it)):

                    if int(next(it)):

                        return "I"

                    else:

                        return "K"

                else:

                    if int(next(it)):

                        return "C"

                    else:

                        if int(next(it)):

                            return "F"

                        else:

                            if int(next(it)):

                                return "."

                            else:

                                return "H"

            else:

                if int(next(it)):

                    if int(next(it)):

                        if int(next(it)):

                            return "T"

                        else:

                            return "S"

                    else:

                        if int(next(it)):

                            return "O"

                        else:

                            return "L"

                else:

                    if int(next(it)):

                        return "D"

                    else:

                        if int(next(it)):

                            if int(next(it)):

                                return ","

                            else:

                                return "W"

                        else:

                            if int(next(it)):

                                return "?"

                            else:

                                return "'"

    def main():

        for line in stdin:

            it = iter("".join(code_table[c] for c in line.rstrip("\r\n")))

            try:

                while 1:

                    print(decrypt(it), end="")

            except StopIteration:

                print()

        exit()

    if __name__ == "__main__":

        main()


problem_p00111()
