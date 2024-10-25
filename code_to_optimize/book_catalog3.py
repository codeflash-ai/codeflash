from __future__ import annotations

from code_to_optimize.book_catalog import Book


def get_authors(session):
    books = session.query(Book).all()
    _authors = []
    for book in books:
        _authors.append(book.author)
    return sorted(list(set(_authors)), key=lambda x: x.id)
