import React from 'react';

interface DataTableProps {
  items: Array<{ id: number; name: string; value: number }>;
  filterText: string;
  sortBy: 'name' | 'value';
}

export function DataTable({ items, filterText, sortBy }: DataTableProps) {
  // These expensive operations run on every render - should use useMemo
  const filteredItems = items.filter(item =>
    item.name.toLowerCase().includes(filterText.toLowerCase())
  );

  const sortedItems = filteredItems.sort((a, b) => {
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    return a.value - b.value;
  });

  const total = sortedItems.reduce((sum, item) => sum + item.value, 0);

  return (
    <div>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {sortedItems.map(item => (
            <tr key={item.id}>
              <td>{item.name}</td>
              <td>{item.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div>Total: {total}</div>
    </div>
  );
}
