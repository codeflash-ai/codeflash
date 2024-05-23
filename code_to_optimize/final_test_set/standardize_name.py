def standardize_name(street_name):
    standard_street_names = [
        "Brattle St",
        "Mount Auburn St",
        "Massachusetts Ave",
        "Cardinal Medeiros Ave",
        "Hampshire Street",
        "Beacon St",
        "Blake St",
        "Beech St",
        "Garden St",
    ]

    # Exact match:
    if street_name in standard_street_names:
        return street_name

    # Different case:
    lower_name = street_name.lower()
    for street in standard_street_names:
        if lower_name == street.lower():
            return street

    # "Ave." and "Avenue" are possible synonyms of "Ave":
    parts = street_name.split()
    if parts[-1].lower() in ("ave.", "avenue"):
        parts[-1] = "Ave"
        fixed_street_name = " ".join(parts)
        return standardize_name(fixed_street_name)

    # "St." and "Street" are possible synonyms of "St":
    if parts[-1].lower() in ("st.", "street"):
        parts[-1] = "St"
        fixed_street_name = " ".join(parts)
        return standardize_name(fixed_street_name)

    raise ValueError(f"Unknown street {street_name}")
