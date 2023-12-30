import ast


def detect_sqlalchemy_session(code):
    """This is just a hack right now, replace it with a better implementation.
    The idea is that if an object quacks like a duck and walks like a duck, it is a duck.
    So if a variable is being using with methods that belong to session object, then the object is a session
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for i, arg in enumerate(node.args.args):
                if arg.arg == "session":
                    # Get rid of this dirty hack, this will only override variables named 'session'
                    return i
    return None


if __name__ == "__main__":
    _code = """def get_authors(session, _session):
    books = session.query(Book).all()
    _authors = []
    for book in books:
        _authors.append(book.author)
    return sorted(list(set(_authors)), key=lambda x: x.id)"""

    print(detect_sqlalchemy_session(_code))
