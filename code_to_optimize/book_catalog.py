from time import time

from sqlalchemy import Boolean, Column, ForeignKey, Integer, Text
from sqlalchemy.engine import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

POSTGRES_CONNECTION_STRING = "postgresql://cf_developer:XJcbU37MBYeh4dDK6PTV5n@sqlalchemy-experiments.postgres.database.azure.com:5432/postgres"
Base = declarative_base()


class Author(Base):
    __tablename__ = "authors"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey("authors.id"), nullable=False)
    is_bestseller = Column(Boolean, default=False)

    author = relationship("Author", backref="books")


def init_table():
    engine = create_engine(POSTGRES_CONNECTION_STRING, echo=True)
    session = sessionmaker(bind=engine)()
    for i in range(50):
        author = Author(id=i, name=f"author{i}")
        session.add(author)
    for i in range(100000):
        book = Book(id=i, title=f"book{i}", author_id=i % 50, is_bestseller=i % 2 == 0)
        session.add(book)
    session.commit()

    return session


def get_authors(session: Session):
    books = session.query(Book).all()
    _authors = []
    for book in books:
        _authors.append(book.author)
    return sorted(list(set(_authors)), key=lambda x: x.id)


if __name__ == "__main__":
    # _session = init_table()
    engine = create_engine(POSTGRES_CONNECTION_STRING, echo=True)
    session_factory = sessionmaker(bind=engine)
    _session = session_factory()
    _t = time()
    authors = get_authors(_session)
    print("TIME TAKEN", time() - _t)
    authors_name = list(map(lambda x: x.name, authors))
    print("len(authors_name)", len(authors_name))
    print(set(authors_name))
