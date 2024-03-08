def problem_p02844():
    def sumitrust2019_d():

        n = int(eval(input()))

        s = eval(input())

        ans = 0

        for i in range(1000):

            # 確認するpinの数字の組み合わせと桁の初期化

            # pin = list(str(i).zfill(3))

            current_digit = 0

            pin = f"{i:0>3}"

            search_num = pin[current_digit]

            for num in s:

                # 見つけると桁を右に一つずらす

                # 毎回リストにアクセスするより変数に取り出しておくほうがいいかも。

                # if num == pin[current_digit]:

                if num == search_num:

                    current_digit += 1

                    # indexが3(桁が3→2→1→0)になったときはそのPINは作れる

                    if current_digit == 3:

                        ans += 1

                        break

                    # 見つけ終わるまではチェックする数字を置き換え。

                    elif current_digit < 3:

                        search_num = pin[current_digit]

        print(ans)

    if __name__ == "__main__":

        sumitrust2019_d()


problem_p02844()
