/**
 * DataGrid component - demonstrates expensive computation patterns.
 *
 * Intentionally inefficient patterns:
 * - Multiple expensive .filter/.sort/.map/.reduce chains on every render
 * - Derived state computed inline without useMemo
 * - No React.memo wrapping
 * - Inline functions in event handlers without useCallback
 *
 * Simulates a data-heavy dashboard grid that becomes slow with large datasets.
 */
import React, { useState } from 'react';

export interface DataRow {
  id: number;
  name: string;
  category: string;
  value: number;
  date: string;
  isActive: boolean;
  tags: string[];
  metadata: Record<string, string>;
}

interface ColumnConfig {
  key: keyof DataRow;
  label: string;
  sortable: boolean;
  filterable: boolean;
  width?: number;
}

interface DataGridProps {
  rows: DataRow[];
  columns: ColumnConfig[];
  pageSize: number;
  onRowSelect: (row: DataRow) => void;
  onBulkAction: (ids: number[], action: string) => void;
}

type SortDirection = 'asc' | 'desc';

/**
 * Data grid with filtering, sorting, pagination, and aggregation.
 * Every derived value is computed inline — prime target for useMemo optimization.
 */
export function DataGrid({ rows, columns, pageSize, onRowSelect, onBulkAction }: DataGridProps) {
  const [currentPage, setCurrentPage] = useState(0);
  const [sortColumn, setSortColumn] = useState<keyof DataRow>('name');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  // Inefficient: expensive filter runs every render
  const filteredRows = rows.filter(row => {
    return Object.entries(filters).every(([key, filterValue]) => {
      if (!filterValue) return true;
      const cellValue = String(row[key as keyof DataRow]).toLowerCase();
      return cellValue.includes(filterValue.toLowerCase());
    });
  });

  // Inefficient: expensive sort runs every render
  const sortedRows = [...filteredRows].sort((a, b) => {
    const aVal = a[sortColumn];
    const bVal = b[sortColumn];
    const direction = sortDirection === 'asc' ? 1 : -1;

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return aVal.localeCompare(bVal) * direction;
    }
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return (aVal - bVal) * direction;
    }
    return 0;
  });

  // Inefficient: pagination computed every render
  const totalPages = Math.ceil(sortedRows.length / pageSize);
  const paginatedRows = sortedRows.slice(
    currentPage * pageSize,
    (currentPage + 1) * pageSize,
  );

  // Inefficient: aggregation computed every render
  const aggregations = {
    totalCount: filteredRows.length,
    activeCount: filteredRows.filter(r => r.isActive).length,
    totalValue: filteredRows.reduce((sum, r) => sum + r.value, 0),
    avgValue: filteredRows.length > 0
      ? filteredRows.reduce((sum, r) => sum + r.value, 0) / filteredRows.length
      : 0,
    categoryBreakdown: filteredRows.reduce(
      (acc, r) => {
        acc[r.category] = (acc[r.category] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    ),
    topCategories: Object.entries(
      filteredRows.reduce(
        (acc, r) => {
          acc[r.category] = (acc[r.category] || 0) + r.value;
          return acc;
        },
        {} as Record<string, number>,
      ),
    )
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5),
    uniqueTags: Array.from(new Set(filteredRows.flatMap(r => r.tags))).sort(),
  };

  // Inefficient: function recreated every render
  const handleSort = (column: keyof DataRow) => {
    if (sortColumn === column) {
      setSortDirection(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  // Inefficient: function recreated every render
  const handleFilterChange = (column: string, value: string) => {
    setFilters(prev => ({ ...prev, [column]: value }));
    setCurrentPage(0);
  };

  // Inefficient: function recreated every render
  const toggleRowSelection = (id: number) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Inefficient: function recreated every render
  const toggleSelectAll = () => {
    if (selectedIds.size === paginatedRows.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(paginatedRows.map(r => r.id)));
    }
  };

  return (
    <div>
      {/* Inefficient: inline style */}
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px' }}>
        <span>
          {aggregations.totalCount} rows | {aggregations.activeCount} active |
          Total value: ${aggregations.totalValue.toFixed(2)} |
          Avg: ${aggregations.avgValue.toFixed(2)}
        </span>
        {selectedIds.size > 0 && (
          <div style={{ display: 'flex', gap: '4px' }}>
            <button onClick={() => onBulkAction(Array.from(selectedIds), 'archive')}>
              Archive ({selectedIds.size})
            </button>
            <button onClick={() => onBulkAction(Array.from(selectedIds), 'delete')}>
              Delete ({selectedIds.size})
            </button>
          </div>
        )}
      </div>

      {/* Inefficient: inline style */}
      <div style={{ padding: '4px 8px', fontSize: '12px', color: '#666' }}>
        Top categories: {aggregations.topCategories.map(([cat, val]) => `${cat}: $${val}`).join(', ')}
        {' | '}
        Tags: {aggregations.uniqueTags.slice(0, 10).join(', ')}
        {aggregations.uniqueTags.length > 10 && ` +${aggregations.uniqueTags.length - 10} more`}
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>
              <input
                type="checkbox"
                checked={selectedIds.size === paginatedRows.length && paginatedRows.length > 0}
                onChange={toggleSelectAll}
              />
            </th>
            {columns.map(col => (
              <th
                key={String(col.key)}
                style={{ width: col.width, cursor: col.sortable ? 'pointer' : 'default' }}
                onClick={() => col.sortable && handleSort(col.key)}
              >
                {col.label}
                {sortColumn === col.key && (sortDirection === 'asc' ? ' ▲' : ' ▼')}
              </th>
            ))}
          </tr>
          <tr>
            <th />
            {columns.map(col => (
              <th key={`filter-${String(col.key)}`}>
                {col.filterable && (
                  <input
                    type="text"
                    placeholder={`Filter ${col.label}...`}
                    value={filters[String(col.key)] || ''}
                    onChange={e => handleFilterChange(String(col.key), e.target.value)}
                    style={{ width: '100%', fontSize: '11px' }}
                  />
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paginatedRows.map(row => (
            <tr
              key={row.id}
              onClick={() => onRowSelect(row)}
              style={{
                cursor: 'pointer',
                background: selectedIds.has(row.id) ? '#e3f2fd' : 'transparent',
                opacity: row.isActive ? 1 : 0.5,
              }}
            >
              <td>
                <input
                  type="checkbox"
                  checked={selectedIds.has(row.id)}
                  onChange={() => toggleRowSelection(row.id)}
                  onClick={e => e.stopPropagation()}
                />
              </td>
              {columns.map(col => (
                <td key={`${row.id}-${String(col.key)}`}>
                  {Array.isArray(row[col.key])
                    ? (row[col.key] as string[]).join(', ')
                    : String(row[col.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Inefficient: inline style */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '4px', padding: '8px' }}>
        <button disabled={currentPage === 0} onClick={() => setCurrentPage(0)}>
          First
        </button>
        <button disabled={currentPage === 0} onClick={() => setCurrentPage(p => p - 1)}>
          Prev
        </button>
        <span>
          Page {currentPage + 1} of {totalPages}
        </span>
        <button
          disabled={currentPage >= totalPages - 1}
          onClick={() => setCurrentPage(p => p + 1)}
        >
          Next
        </button>
        <button
          disabled={currentPage >= totalPages - 1}
          onClick={() => setCurrentPage(totalPages - 1)}
        >
          Last
        </button>
      </div>
    </div>
  );
}
