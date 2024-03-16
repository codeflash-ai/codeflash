def problem_p03547(input_data):
    #!/usr/bin/env python3

    # ABC78 A

    x, y = list(input_data.split())

    if x < y:

        return "<"

    elif x > y:

        return ">"

    elif x == y:

        return "="
