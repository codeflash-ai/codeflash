def problem_p02392(input_data):
    l = input_data.split()

    if int(l[0]) < int(l[1]) and int(l[1]) < int(l[2]):

        return "Yes"

    else:

        return "No"
