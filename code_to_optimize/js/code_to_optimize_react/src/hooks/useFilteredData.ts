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
import { useState, useMemo, useCallback } from 'react';

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

  // Optimized: memoize lower-cased query to avoid repeated toLowerCase calls
  const lowerQuery = useMemo(() => searchQuery.toLowerCase(), [searchQuery]);

  // Optimized: filter is memoized and implemented with a fast for-loop
  const filteredItems = useMemo(() => {
    // When there's no query we still return a new array (like data.filter would)
    if (!lowerQuery) return data.slice();

    const sf = options.searchField;
    const result: T[] = [];
    for (let i = 0, len = data.length; i < len; i++) {
      const item = data[i];
      const fieldValue = String(item[sf] || '');
      if (fieldValue.toLowerCase().includes(lowerQuery)) {
        result.push(item);
      }
    }
    return result;
  }, [data, lowerQuery, options.searchField]);

  // Optimized: sort is memoized; we sort a shallow copy to avoid mutating shared arrays
  const sortedItems = useMemo(() => {
    const arr = filteredItems.slice();
    const dir = sortDirection === 'asc' ? 1 : -1;
    const sf = sortField;

    // Use a single comparator with simple type checks
    arr.sort((a, b) => {
      const aVal = a[sf];
      const bVal = b[sf];

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return (aVal as string).localeCompare(bVal as string) * dir;
      }
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return ((aVal as number) - (bVal as number)) * dir;
      }
      return 0;
    });

    return arr;
  }, [filteredItems, sortField, sortDirection]);

  // Optimized: stable callback for toggling sort direction
  const toggleSortDirection = useCallback(() => {
    setSortDirection(d => (d === 'asc' ? 'desc' : 'asc'));
  }, []);

  // Optimized: memoize returned object to avoid creating a new reference each render
  return useMemo<FilteredDataResult<T>>(
    () => ({
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
    }),
    [
      sortedItems,
      searchQuery,
      setSearchQuery,
      sortField,
      sortDirection,
      setSortField,
      toggleSortDirection,
      data.length,
    ],
  );
}
