def problem_p03657(input_data):
    return (eval(input_data.replace(*" *")) % 3 % 2 * "Imp" or "P") + "ossible"
