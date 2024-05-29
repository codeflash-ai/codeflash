from typing import Generator

import pytest
from sqlalchemy import create_engine, update, delete, Engine
from sqlalchemy.orm import sessionmaker, Session

from code_to_optimize.book_catalog import get_authors, Author, Book

POSTGRES_CONNECTION_STRING = ("postgresql://cf_developer:XJcbU37MBYeh4dDK6PTV5n@sqlalchemy-experiments.postgres"
                              ".database.azure.com:5432/postgres")


@pytest.fixture(scope='module')
def engine() -> Engine:
    return create_engine(POSTGRES_CONNECTION_STRING)


@pytest.fixture(scope='module')
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)


@pytest.fixture(scope='function')
def session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = session_factory()
    yield session
    session.rollback()
    session.close()


def test_get_authors_basic(session: Session) -> None:
    authors = get_authors(session)
    assert len(authors) == 50, "Should return 50 authors"
    author_names = [author.name for author in authors]
    for i in range(50):
        assert f"author{i}" in author_names, f"author{i} should be in the list of authors"


def test_get_authors_no_books(session: Session) -> None:
    # Simulate no books scenario by temporarily deleting books and rolling back
    session.execute(delete(Book))
    session.rollback()
    authors = get_authors(session)
    assert len(authors) == 0, "Should return no authors when there are no books"


def test_get_authors_all_books_same_author(session: Session) -> None:
    # Simulate all books having the same author and rolling back
    session.execute(update(Book).values(author_id=0))
    session.rollback()
    authors = get_authors(session)
    assert len(authors) == 1, "Should return one author when all books have the same author"
    assert authors[0].name == "author0", "The single author should be author0"


def test_get_authors_duplicate_authors(session: Session) -> None:
    # Ensure there are additional authors and simulate duplicate authors in books
    for i in range(50, 60):
        session.add(Author(id=i, name=f"author{i}"))
    session.flush()  # Use flush to apply changes without committing

    session.execute(update(Book).where(Book.id % 2 == 0).values(author_id=(50 + (Book.id % 10))))
    session.rollback()
    authors = get_authors(session)
    assert len(authors) == 60, "Should return 60 authors including duplicates"
