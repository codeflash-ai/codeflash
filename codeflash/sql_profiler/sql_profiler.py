# import threading
# import time
# from functools import wraps
# from typing import Any, Callable

# from psycopg2.extensions import connection, cursor

# _local = threading.local()


# class ProfilingCursor:
#     def __init__(self, real_cursor: cursor) -> None:
#         self._cursor = real_cursor

#     def execute(self, sql, params=None) -> Any:
#         start = time.perf_counter()
#         result = self._cursor.execute(sql, params or ())
#         duration = time.perf_counter() - start
#         _local.queries.append((sql, params, duration))
#         return result

#     def executemany(self, sql, seq_of_params):
#         start = time.perf_counter()
#         result = self._cursor.executemany(sql, seq_of_params)
#         duration = time.perf_counter() - start
#         _local.queries.append((sql, seq_of_params, duration))
#         return result

#     def __getattr__(self, name):
#         return getattr(self._cursor, name)


# class ProfilingConnection:
#     def __init__(self, real_conn: connection) -> None:
#         self._conn = real_conn

#     def cursor(self, *args, **kwargs) -> ProfilingCursor:
#         return ProfilingCursor(self._conn.cursor(*args, **kwargs))

#     def __getattr__(self, name: str) -> Any:
#         return getattr(self._conn, name)


# def sql_profiler(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args, **kwargs) -> Any:
#         _local.queries = []
#         conn_patches = []

#         # Patch psycopg2 (PostgreSQL)
#         try:
#             import psycopg2

#             original_connect = psycopg2.connect

#             def patched_connect(*a, **kw) -> ProfilingConnection:
#                 real_conn = original_connect(*a, **kw)
#                 return ProfilingConnection(real_conn)

#             psycopg2.connect = patched_connect
#             conn_patches.append(("psycopg2", psycopg2, "connect", original_connect))
#         except ImportError:
#             pass

#         # Patch sqlite3
#         try:
#             import sqlite3

#             original_connect = sqlite3.connect

#             def patched_connect(*a, **kw) -> ProfilingConnection:
#                 real_conn = original_connect(*a, **kw)
#                 print(f"real_conn: {real_conn}")
#                 return ProfilingConnection(real_conn)

#             sqlite3.connect = patched_connect
#             conn_patches.append(("sqlite3", sqlite3, "connect", original_connect))
#         except ImportError:
#             pass

#         # Patch Django's database backend
#         try:
#             import django.db.backends.utils as db_utils  # type: ignore # noqa

#             original_execute = db_utils.CursorWrapper.execute
#             original_executemany = db_utils.CursorWrapper.executemany

#             def profiling_execute(self, sql, params=None) -> Any:
#                 start = time.perf_counter()
#                 result = original_execute(self, sql, params)
#                 duration = time.perf_counter() - start
#                 _local.queries.append((sql, params, duration))
#                 return result

#             def profiling_executemany(self, sql, param_list) -> Any:
#                 start = time.perf_counter()
#                 result = original_executemany(self, sql, param_list)
#                 duration = time.perf_counter() - start
#                 _local.queries.append((sql, param_list, duration))
#                 return result

#             db_utils.CursorWrapper.execute = profiling_execute
#             db_utils.CursorWrapper.executemany = profiling_executemany
#             conn_patches.append(("django.execute", db_utils.CursorWrapper, "execute", original_execute))
#             conn_patches.append(("django.executemany", db_utils.CursorWrapper, "executemany", original_executemany))
#         except ImportError:
#             pass

#         # Run the actual function
#         wall_start = time.perf_counter()
#         try:
#             return func(*args, **kwargs)
#         finally:
#             wall_end = time.perf_counter()
#             total_wall = wall_end - wall_start
#             total_sql = sum(q[2] for q in _local.queries)

#             print("\n📊 SQL PROFILER REPORT")
#             print(f"Total wall time:    {total_wall:.4f}s")
#             print(f"Total SQL time:     {total_sql:.4f}s")
#             print(f"Total queries:      {len(_local.queries)}")
#             print("-" * 60)
#             for i, (sql, params, dur) in enumerate(_local.queries, 1):
#                 print(f"[{i}] {dur:.4f}s: {sql}")
#                 if params:
#                     print(f"    params: {params}")
#             print("-" * 60)

#             # Restore original connection functions
#             for _, module, attr, orig in conn_patches:
#                 setattr(module, attr, orig)

#     return wrapper
