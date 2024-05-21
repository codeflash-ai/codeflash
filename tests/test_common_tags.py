from codeflash.result.common_tags import find_common_tags


def test_common_tags_1():
    articles_1 = [
        {"title": "Article 1", "tags": ["Python", "AI", "ML"]},
        {"title": "Article 2", "tags": ["Python", "Data Science", "AI"]},
        {"title": "Article 3", "tags": ["Python", "AI", "Big Data"]},
    ]

    expected = set(["Python", "AI"])

    assert find_common_tags(articles_1) == expected

    articles_2 = [
        {"title": "Article 1", "tags": ["Python", "AI", "ML"]},
        {"title": "Article 2", "tags": ["Python", "Data Science", "AI"]},
        {"title": "Article 3", "tags": ["Python", "AI", "Big Data"]},
        {"title": "Article 4", "tags": ["Python", "AI", "ML"]},
    ]

    assert find_common_tags(articles_2) == expected
