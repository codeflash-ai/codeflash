def problem_p04030(input_data):
    def main():

        s = eval(input_data)

        res = ""

        for i in s:

            if i == "B":

                res = res[:-1]

            else:

                res += i

        return res

    if __name__ == "__main__":

        main()
