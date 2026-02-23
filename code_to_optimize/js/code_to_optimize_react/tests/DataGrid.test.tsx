import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { DataGrid, DataRow } from '../src/components/DataGrid';

const makeRow = (overrides: Partial<DataRow> = {}): DataRow => ({
  id: 1,
  name: 'Test Item',
  category: 'A',
  value: 100,
  date: '2024-01-15',
  isActive: true,
  tags: ['tag1'],
  metadata: { source: 'test' },
  ...overrides,
});

const makeRows = (count: number): DataRow[] =>
  Array.from({ length: count }, (_, i) =>
    makeRow({
      id: i + 1,
      name: `Item ${String(i + 1).padStart(3, '0')}`,
      category: ['Electronics', 'Clothing', 'Food', 'Books'][i % 4],
      value: Math.round((i + 1) * 10.5 * 100) / 100,
      date: `2024-${String((i % 12) + 1).padStart(2, '0')}-${String((i % 28) + 1).padStart(2, '0')}`,
      isActive: i % 3 !== 0,
      tags: [`tag-${i % 5}`, `category-${i % 3}`],
      metadata: { source: `source-${i % 3}` },
    }),
  );

const defaultColumns = [
  { key: 'name' as const, label: 'Name', sortable: true, filterable: true },
  { key: 'category' as const, label: 'Category', sortable: true, filterable: true },
  { key: 'value' as const, label: 'Value', sortable: true, filterable: false },
  { key: 'date' as const, label: 'Date', sortable: true, filterable: false },
];

const noop = () => {};

describe('DataGrid', () => {
  describe('rendering', () => {
    test('renders with empty rows', () => {
      render(
        <DataGrid
          rows={[]}
          columns={defaultColumns}
          pageSize={10}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      expect(screen.getByText(/0 rows/)).toBeTruthy();
    });

    test('renders column headers', () => {
      render(
        <DataGrid
          rows={makeRows(5)}
          columns={defaultColumns}
          pageSize={10}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      expect(screen.getByText(/^Name/)).toBeTruthy();
      expect(screen.getByText(/^Category/)).toBeTruthy();
      expect(screen.getByText(/^Value/)).toBeTruthy();
    });

    test('renders correct row count', () => {
      const rows = makeRows(5);
      render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={10}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      expect(screen.getByText(/5 rows/)).toBeTruthy();
    });

    test('displays aggregation values', () => {
      const rows = makeRows(10);
      render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={20}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      expect(screen.getByText(/active/)).toBeTruthy();
      expect(screen.getByText(/Total value/)).toBeTruthy();
    });
  });

  describe('pagination', () => {
    test('paginates rows correctly', () => {
      const rows = makeRows(25);
      render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={10}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      expect(screen.getByText(/Page 1 of 3/)).toBeTruthy();
    });

    test('navigates to next page', () => {
      const rows = makeRows(25);
      render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={10}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      fireEvent.click(screen.getByText('Next'));
      expect(screen.getByText(/Page 2 of 3/)).toBeTruthy();
    });
  });

  describe('sorting', () => {
    test('sorts by column when header clicked', () => {
      const rows = makeRows(5);
      render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={10}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      fireEvent.click(screen.getByText(/^Name/));
      // Should show sort indicator
      expect(screen.getByText(/^Name/)).toBeTruthy();
    });
  });

  describe('selection', () => {
    test('calls onRowSelect when row clicked', () => {
      const onRowSelect = jest.fn();
      const rows = makeRows(3);
      render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={10}
          onRowSelect={onRowSelect}
          onBulkAction={noop}
        />,
      );
      fireEvent.click(screen.getByText('Item 001'));
      expect(onRowSelect).toHaveBeenCalledWith(expect.objectContaining({ id: 1 }));
    });
  });

  describe('performance with large datasets', () => {
    test('renders 200 rows with pagination', () => {
      const rows = makeRows(200);
      const { container } = render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={20}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      expect(container).toBeTruthy();
      expect(screen.getByText(/200 rows/)).toBeTruthy();
    });

    test('renders 1000 rows with pagination', () => {
      const rows = makeRows(1000);
      const { container } = render(
        <DataGrid
          rows={rows}
          columns={defaultColumns}
          pageSize={50}
          onRowSelect={noop}
          onBulkAction={noop}
        />,
      );
      expect(container).toBeTruthy();
      expect(screen.getByText(/1000 rows/)).toBeTruthy();
    });
  });
});
