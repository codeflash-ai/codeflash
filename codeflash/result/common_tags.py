from __future__ import annotations


def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0].get("tags", [])
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article.get("tags", [])]
    return set(common_tags)
