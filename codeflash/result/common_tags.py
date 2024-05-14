def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        common_tags &= set(article["tags"])  # using set intersection
    return common_tags
