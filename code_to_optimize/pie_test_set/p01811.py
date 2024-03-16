def problem_p01811(input_data):
    #!/usr/bin/python

    if __name__ == "__main__":

        S = input_data

        c = ["A", "B", "C"]

        while True:

            if len(S) <= 3:

                return "Yes" if S == "ABC" else "No"

                break

            T = S.strip().split("ABC")

            if len(T) == 1:

                return "No"

                break

            P = "".join(T)

            cnt = 0

            for x in c:

                if x in P:

                    cnt += 1

                else:

                    S = x.join(T)

            if cnt != 2:

                return "No"

                break
