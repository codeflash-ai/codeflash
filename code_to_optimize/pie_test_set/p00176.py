def problem_p00176():
    clr = ["black", "blue", "lime", "aqua", "red", "fuchsia", "yellow", "white"]

    hex = ["000000", "0000ff", "00ff00", "00ffff", "ff0000", "ff00ff", "ffff00", "ffffff"]

    def L(s1, s2):

        return (
            (int(s1[:2], 16) - int(s2[:2], 16)) ** 2
            + (int(s1[2:4], 16) - int(s2[2:4], 16)) ** 2
            + (int(s1[4:], 16) - int(s2[4:], 16)) ** 2
        ) ** 0.5

    def nn(s):

        mn = mni = 99999

        for i in range(8):

            if mn > L(hex[i], s):

                mn = L(hex[i], s)
                mni = i

        return mni

    while 1:

        rgb = input()

        if rgb == "0":
            break

        print(clr[nn(rgb[1:])])


problem_p00176()
