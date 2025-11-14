from __future__ import annotations


def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0].get("tags", []))
    for article in articles[1:]:
        common_tags.intersection_update(article.get("tags", []))
        if not common_tags:  # Early exit if no common tags left
            break
    return common_tags
