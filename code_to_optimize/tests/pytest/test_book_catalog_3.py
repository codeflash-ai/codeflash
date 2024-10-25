from typing import Generator

import pytest
from sqlalchemy import Engine, create_engine, delete, update
from sqlalchemy.orm import Session, sessionmaker

from code_to_optimize.book_catalog import Author, Book, get_top_author

POSTGRES_CONNECTION_STRING = (
    "postgresql://cf_developer:XJcbU37MBYeh4dDK6PTV5n@sqlalchemy-experiments.postgres"
    ".database.azure.com:5432/postgres"
)


def test_get_top_author():
    engine: Engine = create_engine(POSTGRES_CONNECTION_STRING, echo=True)
    session_factory: sessionmaker[Session] = sessionmaker(bind=engine)
    session: Session = session_factory()
    authors = session.query(Author).all()
    top_author = get_top_author(authors)
    assert top_author.id == 0
    assert top_author.name == "author0"
