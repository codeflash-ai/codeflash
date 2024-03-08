def problem_p00511():
    if __name__ == "__main__":

        opcode = []  # ?????????????¨???¶??¨

        operand = []  # ?????´????¢??????????????¨???¶??¨

        # ??????????????\???

        while True:

            data = eval(input())

            if data == "=":

                break

            elif data == "+" or data == "-" or data == "*" or data == "/":

                opcode.append(data)

            else:

                if len(operand) == 0:

                    operand.append(int(data))

                else:

                    op = opcode.pop()

                    operand1 = operand.pop()

                    operand2 = int(data)

                    if op == "+":

                        operand.append(operand1 + operand2)

                    elif op == "-":

                        operand.append(operand1 - operand2)

                    elif op == "*":

                        operand.append(operand1 * operand2)

                    elif op == "/":

                        operand.append(operand1 // operand2)

        # ???????????¨???

        print(("{0}".format(operand.pop())))


problem_p00511()
