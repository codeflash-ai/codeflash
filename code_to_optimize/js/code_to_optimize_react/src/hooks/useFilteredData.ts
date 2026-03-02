/**
 * useFilteredData hook - demonstrates hook optimization patterns.
 *
 * Intentionally inefficient patterns:
 * - Returns new object reference on every call (breaks shallow comparison)
 * - Filter/sort computed without useMemo
 * - Callbacks recreated without useCallback
 * - Derived state computed inline
 *
 * Custom hooks that return unstable references are a common source of
 * unnecessary re-renders in components that consume them.
 */
import { useState } from 'react';

interface FilterOptions<T> {
  searchField: keyof T;
  sortField: keyof T;
  sortDirection: 'asc' | 'desc';
}

interface FilteredDataResult<T> {
  filteredItems: T[];
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  sortField: keyof T;
  sortDirection: 'asc' | 'desc';
  setSortField: (field: keyof T) => void;
  toggleSortDirection: () => void;
  totalCount: number;
  filteredCount: number;
  isEmpty: boolean;
}

/**
 * Hook for filtering and sorting data arrays.
 * Returns unstable references — consumers re-render on every parent render.
 */
export function useFilteredData<T extends Record<string, unknown>>(
  data: T[],
  options: FilterOptions<T>,
): FilteredDataResult<T> {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<keyof T>(options.sortField);
  const [sortDirection, setSortDirection] = useState(options.sortDirection);

  // Inefficient: filter runs every call
  const filteredItems = data.filter(item => {
    if (!searchQuery) return true;
    const fieldValue = String(item[options.searchField] || '');
    return fieldValue.toLowerCase().includes(searchQuery.toLowerCase());
  });

  // Inefficient: sort runs every call
  const sortedItems = filteredItems.sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];
    const dir = sortDirection === 'asc' ? 1 : -1;

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return aVal.localeCompare(bVal) * dir;
    }
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return (aVal - bVal) * dir;
    }
    return 0;
  });

  // Inefficient: function recreated every call
  const toggleSortDirection = () => {
    setSortDirection(d => (d === 'asc' ? 'desc' : 'asc'));
  };

  // Inefficient: returns new object reference every call
  return {
    filteredItems: sortedItems,
    searchQuery,
    setSearchQuery,
    sortField,
    sortDirection,
    setSortField,
    toggleSortDirection,
    totalCount: data.length,
    filteredCount: sortedItems.length,
    isEmpty: sortedItems.length === 0,
  };
}
