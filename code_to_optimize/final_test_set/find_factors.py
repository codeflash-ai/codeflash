def find_factors(product):
    answers = []
    for factor in range(1, product + 1):
        if not product % factor:
            factor2 = int(product / factor)
            answers.append((factor, factor2))
    return answers
