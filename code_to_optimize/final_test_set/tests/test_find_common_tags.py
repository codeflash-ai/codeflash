from code_to_optimize.final_test_set.find_common_tags import find_common_tags


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


def test_empty_article_list():
    articles = []
    expected = set()
    assert find_common_tags(articles) == expected, "Test failed for empty list of articles."


def test_no_common_tags():
    articles = [
        {"tags": ["python", "coding", "tutorial"]},
        {"tags": ["java", "software", "programming"]},
        {"tags": ["javascript", "development", "web"]},
    ]
    expected = set()
    assert find_common_tags(articles) == expected, "Test failed when no tags are common."


def test_all_common_tags():
    articles = [
        {"tags": ["tech", "startups", "innovation"]},
        {"tags": ["tech", "startups", "innovation"]},
        {"tags": ["tech", "startups", "innovation"]},
    ]
    expected = {"tech", "startups", "innovation"}
    assert find_common_tags(articles) == expected, "Test failed when all tags are common."


def test_single_article():
    articles = [{"tags": ["single", "article", "test"]}]
    expected = {"single", "article", "test"}
    assert find_common_tags(articles) == expected, "Test failed for a single article input."
