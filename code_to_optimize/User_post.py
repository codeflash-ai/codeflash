from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String, create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Relationship,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


# Custom base class
class Base(DeclarativeBase):
    pass


engine: Engine = create_engine('sqlite:///example.db')

session_factory = sessionmaker(bind=engine)
session: Session = session_factory()


class User(Base):
    __tablename__: str = 'users'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    posts: Relationship[list[Post]] = relationship("Post", order_by="Post.id", back_populates="user")


class Post(Base):
    __tablename__: str = 'posts'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey('users.id'))
    user: Relationship[User] = relationship("User", back_populates="posts")


Base.metadata.create_all(engine)


def get_user_posts() -> dict[User, list[Post]]:
    users: list[User] = session.query(User).all()  # Query all users
    user_posts: dict[User, list[Post]] = {}
    for u in users:
        user_posts[u] = session.query(Post).filter(Post.user_id == u.id).all()
    return user_posts


# Example usage
for user, posts in get_user_posts().items():
    print(f"User: {user.name}, Posts: {[post.title for post in posts]}")
