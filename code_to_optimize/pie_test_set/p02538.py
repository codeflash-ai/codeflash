def problem_p02538():
    code = r"""
    
    # distutils: language=c++
    
    # distutils: include_dirs=/opt/ac-library
    
    # cython: cdivision=True
    
    from libcpp cimport bool
    
    from libcpp.vector cimport vector
    
    cdef extern from "<atcoder/lazysegtree>" namespace "atcoder" nogil:
    
        cdef cppclass lazy_segtree[S, OP, E, F, mapping, composition, id]:
    
            lazy_segtree(vector[S] v)
    
            void set(int p, S x)
    
            S get(int p)
    
            S prod(int l, int r)
    
            S all_prod()
    
            void apply(int p, F f)
    
            void apply(int l, int r, F f)
    
            int max_right[G](int l)
    
            int min_left[G](int r)
    
    cdef long mod = 998244353
    
    cdef struct s_t:
    
        long v, s
    
    ctypedef long f_t
    
    cdef extern from *:
    
        ctypedef int op_t "myop"
    
        ctypedef int e_t "mye"
    
        ctypedef int m_t "mymapping"
    
        ctypedef int c_t "mycomposition"
    
        ctypedef int i_t "myid"
    
        ctypedef int g_t "myg"
    
        cdef s_t myop(s_t a, s_t b) nogil
    
        cdef s_t mye() nogil
    
        cdef s_t mymapping(f_t f, s_t x) nogil
    
        cdef f_t mycomposition(f_t f, f_t g) nogil
    
        cdef f_t myid() nogil
    
        cdef bool myg(s_t k) nogil
    
    cdef s_t myop(s_t a, s_t b) nogil:
    
        cdef s_t c
    
        c.v = (a.v * b.s + b.v) % mod
    
        c.s = (a.s * b.s) % mod
    
        return c
    
    cdef s_t mye() nogil:
    
        cdef s_t e
    
        e.v = 0
    
        e.s = 1
    
        return e
    
    cdef f_t myid() nogil:
    
        return -1
    
    cdef s_t mymapping(f_t f, s_t x) nogil:
    
        if f == myid(): return x
    
        cdef s_t y
    
        y.v = (x.s - 1) * 443664157 * f % mod
    
        y.s = x.s
    
        return y
    
    cdef f_t mycomposition(f_t f, f_t g) nogil:
    
        if f == myid(): return g
    
        return f
    
    from libc.stdio cimport getchar, printf
    
    cdef inline int read() nogil:
    
        cdef int b, c = 0
    
        while 1:
    
            b = getchar() - 48
    
            if b < 0: return c
    
            c = c * 10 + b
    
    cdef main():
    
        cdef int n = read(), i, l, r, d
    
        cdef s_t a
    
        a.v = 1
    
        a.s = 10
    
        cdef lazy_segtree[s_t, op_t, e_t, f_t, m_t, c_t, i_t] *seg = new lazy_segtree[s_t, op_t, e_t, f_t, m_t, c_t, i_t](vector[s_t](n, a))
    
        for i in range(read()):
    
            l, r, d = read(), read(), read()
    
            seg.apply(l - 1, r, d)
    
            printf('%ld\n', seg.all_prod().v)
    
    main()
    
    """

    import os, sys

    if sys.argv[-1] == "ONLINE_JUDGE":

        open("solve.pyx", "w").write(code)

        os.system("cythonize -i -3 -b solve.pyx")

    import solve


problem_p02538()
