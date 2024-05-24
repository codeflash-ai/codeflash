from __future__ import annotations

from code_to_optimize.book_catalog import (
    POSTGRES_CONNECTION_STRING,
    Author,
    Base,
    Book,
    _session,
    _t,
    authors,
    authors_name,
    engine,
    init_table,
    session_factory,
)


def get_authors(session):
    books = session.query(Book).all()
    _authors = []
    for book in books:
        _authors.append(book.author)
    return sorted(list(set(_authors)), key=lambda x: x.id)
