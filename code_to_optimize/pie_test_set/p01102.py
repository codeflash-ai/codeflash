def problem_p01102():
    while 1:

        s1 = input().split('"')

        if s1[0] == ".":

            break

        s2 = input().split('"')

        cnt1 = 0

        cnt2 = 0

        flag = 0

        if len(s1) != len(s2):

            print("DIFFERENT")

        else:

            i = 0

            while i < len(s1):

                if s1[i] == s2[i]:

                    cnt1 += 1

                elif i % 2 == 0:

                    print("DIFFERENT")

                    flag = 1

                    break

                else:

                    cnt2 += 1

                    if cnt2 > 1:

                        print("DIFFERENT")

                        break

                i += 1

            if flag == 0 and cnt2 == 1:

                print("CLOSE")

            if cnt1 == len(s1):

                print("IDENTICAL")


problem_p01102()
