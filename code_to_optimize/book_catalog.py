from time import time
from typing import List

from sqlalchemy import Boolean, Column, ForeignKey, Integer, Text, func
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.orm.relationships import Relationship

POSTGRES_CONNECTION_STRING: str = ("postgresql://cf_developer:XJcbU37MBYeh4dDK6PTV5n@sqlalchemy-experiments.postgres"
                                   ".database.azure.com:5432/postgres")


class Base(DeclarativeBase):
    pass


class Author(Base):
    __tablename__: str = "authors"

    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(Text, nullable=False)


class Book(Base):
    __tablename__: str = "books"

    id: Column[int] = Column(Integer, primary_key=True)
    title: Column[str] = Column(Text, nullable=False)
    author_id: Column[int] = Column(Integer, ForeignKey("authors.id"), nullable=False)
    is_bestseller: Column[bool] = Column(Boolean, default=False)

    author: Relationship[Author] = relationship("Author", backref="books")


def init_table() -> Session:
    catalog_engine: Engine = create_engine(POSTGRES_CONNECTION_STRING, echo=True)
    session: Session = sessionmaker(bind=catalog_engine)()
    i: int
    for i in range(50):
        author: Author = Author(id=i, name=f"author{i}")
        session.add(author)
    for i in range(100000):
        book: Book = Book(id=i, title=f"book{i}", author_id=i % 50, is_bestseller=i % 2 == 0)
        session.add(book)
    session.commit()

    return session


def get_authors(books: list[Book]) -> list[Author]:
    _authors: list[Author] = []
    book: Book
    for book in books:
        _authors.append(book.author)
    return sorted(
        list(set(_authors)),
        key=lambda x: x.id,
    )

def get_authors2(num_authors) -> list[Author]:
    engine: Engine = create_engine(POSTGRES_CONNECTION_STRING, echo=True)
    session_factory: sessionmaker[Session] = sessionmaker(bind=engine)
    session: Session = session_factory()
    books: list[Book] = session.query(Book).all()
    _authors: list[Author] = []
    book: Book
    for book in books:
        _authors.append(book.author)
    return sorted(
        list(set(_authors)),
        key=lambda x: x.id,
    )[:num_authors]


def get_top_author(authors: List[Author]) -> Author:
    engine: Engine = create_engine(POSTGRES_CONNECTION_STRING, echo=True)
    session_factory: sessionmaker[Session] = sessionmaker(bind=engine)
    session: Session = session_factory()

    # Step 1: Initialize variables to keep track of the author with the maximum bestsellers
    max_bestsellers = 0
    top_author = None

    # Step 2: Iterate over each author to count their bestsellers
    for author in authors:
        bestseller_count = (
            session.query(func.count(Book.id))
            .filter(Book.author_id == author.id, Book.is_bestseller == True)
            .scalar()
        )

        # Step 3: Update the author with the maximum bestsellers
        if bestseller_count > max_bestsellers:
            max_bestsellers = bestseller_count
            top_author = author

    return top_author


if __name__ == "__main__":
    engine: Engine = create_engine(POSTGRES_CONNECTION_STRING, echo=True)
    session_factory: sessionmaker[Session] = sessionmaker(bind=engine)
    _session: Session = session_factory()
    _t: float = time()
    authors: list[Author] = get_authors(_session)
    print("TIME TAKEN", time() - _t)
    authors_name = list(map(lambda x: x.name, authors))
    print("len(authors_name)", len(authors_name))
    print(set(authors_name))
