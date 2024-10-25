from typing import Generator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from code_to_optimize.book_catalog import Book, get_authors

POSTGRES_CONNECTION_STRING = (
    "postgresql://cf_developer:XJcbU37MBYeh4dDK6PTV5n@sqlalchemy-experiments.postgres"
    ".database.azure.com:5432/postgres"
)


@pytest.fixture(scope="module")
def engine() -> Engine:
    return create_engine(POSTGRES_CONNECTION_STRING)


@pytest.fixture(scope="module")
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)


@pytest.fixture(scope="function")
def session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = session_factory()
    yield session
    session.rollback()
    session.close()


def test_get_authors_basic(session: Session) -> None:
    books: list[Book] = session.query(Book).all()
    authors = get_authors(books)
    assert len(authors) == 50, "Should return 50 authors"
    author_names = [author.name for author in authors]
    for i in range(50):
        assert f"author{i}" in author_names, f"author{i} should be in the list of authors"
