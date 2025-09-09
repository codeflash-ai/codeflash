from __future__ import annotations

from codeflash.code_utils.line_profile_utils import LineProfilerDecorator


@codeflash_line_profile
def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0]["tags"]
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article["tags"]]
    return set(common_tags)
