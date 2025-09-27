# ORIGINAL SLOW CODE


def find_common_tags(news_articles) -> set[str]:
    if not news_articles:
        return set()

    common_tags = news_articles[0]["tags"]
    for article in news_articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article.get("tags", [])]
    return set(common_tags)


# OPTIMIZATION 1: Does not win  - Is 33x faster


def find_common_tags(news_articles) -> set[str]:
    if not news_articles:
        return set()

    common_tags = set(news_articles[0]["tags"])
    for article in news_articles[1:]:
        common_tags &= set(article.get("tags", []))
    return common_tags


# OPTIMIZATION 2 : Winning Optimization - Is 90x faster


def find_common_tags(news_articles) -> set[str]:
    if not news_articles:
        return set()

    common_tags = set(news_articles[0]["tags"])
    for article in news_articles[1:]:
        common_tags.intersection_update(article.get("tags", []))
    return common_tags
