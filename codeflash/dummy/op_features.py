import psycopg2
from psycopg2.extensions import connection


def get_db_connection() -> connection:
    # let's say those are the real db configs, pytest patch should be used instead
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="your_database",
        user="your_user",
        password="your_password",  # noqa: S106
    )


def get_all_optimization_features() -> list:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM optimization_features")
            return cursor.fetchall()
    finally:
        conn.close()
