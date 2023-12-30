import pickle

from cli.code_to_optimize.pig_latin import pig_latin


def log_test_values(values, test_name):
    with open(f"/tmp/test_return_values.bin", "ab") as f:
        return_bytes = pickle.dumps(values)
        _test_name = f"{test_name}".encode("ascii")
        f.write(len(_test_name).to_bytes(4, byteorder="big"))
        f.write(_test_name)
        f.write(len(return_bytes).to_bytes(4, byteorder="big"))
        f.write(return_bytes)


def test_pig_latin_vowel():
    global log_test_values
    log_test_values(pig_latin("apple"), "pig_latin_test_pig_latin_vowel_0")
    log_test_values(pig_latin("elephant"), "pig_latin_test_pig_latin_vowel_1")


def test_pig_latin_single_consonant():
    log_test_values(pig_latin("dog"), "pig_latin_test_pig_latin_single_consonant_0")
    log_test_values(pig_latin("cat"), "pig_latin_test_pig_latin_single_consonant_1")


def test_pig_latin_multiple_consonants():
    log_test_values(pig_latin("string"), "pig_latin_test_pig_latin_multiple_consonants_0")
    log_test_values(pig_latin("glove"), "pig_latin_test_pig_latin_multiple_consonants_1")


def test_pig_latin_capital_letters():
    log_test_values(pig_latin("Hello"), "pig_latin_test_pig_latin_capital_letters_0")
    log_test_values(pig_latin("WoRlD"), "pig_latin_test_pig_latin_capital_letters_1")


def test_pig_latin_multiple_words():
    log_test_values(pig_latin("The quick brown fox"), "pig_latin_test_pig_latin_multiple_words_0")
    log_test_values(
        pig_latin("Python is a fun language"), "pig_latin_test_pig_latin_multiple_words_1"
    )


def test_pig_latin_empty_input():
    log_test_values(pig_latin(""), "pig_latin_test_pig_latin_empty_input_0")


def test_pig_latin_spaces_input():
    log_test_values(pig_latin("   "), "pig_latin_test_pig_latin_spaces_input_0")


def test_pig_latin_non_alphabetic():
    log_test_values(pig_latin("123"), "pig_latin_test_pig_latin_non_alphabetic_0")
    log_test_values(pig_latin("Hello, world!"), "pig_latin_test_pig_latin_non_alphabetic_1")


def test_pig_latin_non_ascii():
    log_test_values(pig_latin("café"), "pig_latin_test_pig_latin_non_ascii_0")
    log_test_values(pig_latin("über"), "pig_latin_test_pig_latin_non_ascii_1")


def test_pig_latin_hyphenated_words():
    log_test_values(pig_latin("sister-in-law"), "pig_latin_test_pig_latin_hyphenated_words_0")
    log_test_values(pig_latin("self-driving car"), "pig_latin_test_pig_latin_hyphenated_words_1")


def test_pig_latin_contractions():
    log_test_values(pig_latin("can't"), "pig_latin_test_pig_latin_contractions_0")
    log_test_values(pig_latin("I'm"), "pig_latin_test_pig_latin_contractions_1")


def test_pig_latin_apostrophes():
    log_test_values(pig_latin("don't"), "pig_latin_test_pig_latin_apostrophes_0")
    log_test_values(pig_latin("rock 'n' roll"), "pig_latin_test_pig_latin_apostrophes_1")


def test_pig_latin_non_letter():
    log_test_values(pig_latin("123"), "pig_latin_test_pig_latin_non_letter_0")
    log_test_values(pig_latin("Hello, world!"), "pig_latin_test_pig_latin_non_letter_1")
