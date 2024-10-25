from __future__ import annotations

from typing import Any, cast

from _typeshed import SupportsDunderGT, SupportsDunderLT
from sqlalchemy.orm import Session

from code_to_optimize.book_catalog import Author, Book


def get_authors(session: Session) -> list[Author]:
    books: list[Book] = session.query(Book).all()
    _authors: list[Author] = []
    book: Book
    for book in books:
        _authors.append(book.author)
    return sorted(list(set(_authors)), key=lambda x: cast(SupportsDunderLT[Any] | SupportsDunderGT[Any], x.id))
