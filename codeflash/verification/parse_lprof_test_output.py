"""
This software is OSI Certified Open Source Software.
OSI Certified is a certification mark of the Open Source Initiative.

Copyright (c) 2008, Enthought, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* Neither the name of Enthought, Inc. nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import linecache
import inspect
from codeflash.code_utils.tabulate import tabulate
import os
import dill as pickle
from pathlib import Path

def show_func(filename, start_lineno, func_name, timings, unit):
    total_hits = sum(t[1] for t in timings)
    total_time = sum(t[2] for t in timings)
    out_table = ""
    table_rows = []
    if total_hits == 0:
        return ''
    scalar = 1
    if os.path.exists(filename):
        out_table+=f'## Function: {func_name}\n'
        # Clear the cache to ensure that we get up-to-date results.
        linecache.clearcache()
        all_lines = linecache.getlines(filename)
        sublines = inspect.getblock(all_lines[start_lineno - 1:])
    out_table+='## Total time: %g s\n' % (total_time * unit)
    # Define minimum column sizes so text fits and usually looks consistent
    default_column_sizes = {
        'hits': 9,
        'time': 12,
        'perhit': 8,
        'percent': 8,
    }
    display = {}
    # Loop over each line to determine better column formatting.
    # Fallback to scientific notation if columns are larger than a threshold.
    for lineno, nhits, time in timings:
        if total_time == 0:  # Happens rarely on empty function
            percent = ''
        else:
            percent = '%5.1f' % (100 * time / total_time)

        time_disp = '%5.1f' % (time * scalar)
        if len(time_disp) > default_column_sizes['time']:
            time_disp = '%5.1g' % (time * scalar)
        perhit_disp = '%5.1f' % (float(time) * scalar / nhits)
        if len(perhit_disp) > default_column_sizes['perhit']:
            perhit_disp = '%5.1g' % (float(time) * scalar / nhits)
        nhits_disp = "%d" % nhits
        if len(nhits_disp) > default_column_sizes['hits']:
            nhits_disp = '%g' % nhits
        display[lineno] = (nhits_disp, time_disp, perhit_disp, percent)
    linenos = range(start_lineno, start_lineno + len(sublines))
    empty = ('', '', '', '')
    table_cols = ('Hits', 'Time', 'Per Hit', '% Time', 'Line Contents')
    for lineno, line in zip(linenos, sublines):
        nhits, time, per_hit, percent = display.get(lineno, empty)
        line_ = line.rstrip('\n').rstrip('\r')
        if 'def' in line_ or nhits!='':
            table_rows.append((nhits, time, per_hit, percent, line_))
    out_table+= tabulate(headers=table_cols,tabular_data=table_rows,tablefmt="pipe")
    out_table+='\n'
    return out_table

def show_text(stats):
    """ Show text for the given timings.
    """
    out_table = ""
    out_table+='# Timer unit: %g s\n' % stats.unit
    stats_order = sorted(stats.timings.items())
    # Show detailed per-line information for each function.
    for (fn, lineno, name), timings in stats_order:
        table_md =show_func(fn, lineno, name, stats.timings[fn, lineno, name], stats.unit)
        out_table+=table_md
    return out_table

def parse_lprof_results(lprofiler_database_file: Path | None) -> str:
    lprofiler_database_file = lprofiler_database_file.with_suffix(".lprof")
    if not lprofiler_database_file.exists():
        return ""
    else:
        with open(lprofiler_database_file,'rb') as f:
            stats = pickle.load(f)
        return show_text(stats), None