def get_authors(session):
    query = session.query(Author).join(Book).distinct(Author.id).order_by(Author.id)
    return query.all()
