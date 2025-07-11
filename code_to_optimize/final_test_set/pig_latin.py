def translate(word):
    vowels = "aeiou"
    if word[0] in vowels:
        return word + "way"
    # Find the index of the first vowel and use slicing to avoid repeated string ops
    for i, letter in enumerate(word):
        if letter in vowels:
            # Use slice and concat
            return word[i:] + word[:i] + "ay"
    # No vowels: treat as all consonants
    return word + "ay"


def pig_latin(text):
    words = text.lower().split()
    translated_words = [translate(word) for word in words]
    return " ".join(translated_words)
