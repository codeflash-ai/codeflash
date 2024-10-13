from code_to_optimize.book_catalog import get_authors2


def test_get_authors_basic() -> None:
    authors = get_authors2(num_authors=10)
    assert len(authors) == 10, "Should return 10 authors"
