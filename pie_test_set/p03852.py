def problem_p03852(input_data):
    c = str.lower(eval(input_data))

    n = ("a", "e", "i", "o", "u")

    if c in n:

        return "vowel"

    else:

        return "consonant"
