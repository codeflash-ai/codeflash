def translate(word):
    vowels = "aeiou"
    if word[0] in vowels:
        return word + "way"
    consonants = ""
    for letter in word:
        if letter not in vowels:
            consonants += letter
        else:
            break
    return word[len(consonants) :] + consonants + "ay"


def pig_latin(text):
    words = text.lower().split()
    translated_words = [translate(word) for word in words]
    return " ".join(translated_words)
