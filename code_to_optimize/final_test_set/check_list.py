def check_user_access(user_ids, check_ids):
    """Check if each ID in check_ids is in the list of user_ids"""
    results = []
    for id in check_ids:
        if id in user_ids:
            results.append(True)
        else:
            results.append(False)
    return results
