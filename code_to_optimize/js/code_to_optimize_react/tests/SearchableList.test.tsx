import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { SearchableList, ListItemCard, ListItem } from '../src/components/SearchableList';

const makeItem = (overrides: Partial<ListItem> = {}): ListItem => ({
  id: 1,
  label: 'Test Item',
  category: 'General',
  timestamp: Date.now(),
  score: 50,
  isFavorite: false,
  ...overrides,
});

const makeItems = (count: number): ListItem[] =>
  Array.from({ length: count }, (_, i) =>
    makeItem({
      id: i + 1,
      label: `Item ${String(i + 1).padStart(3, '0')}`,
      category: ['Work', 'Personal', 'Shopping', 'Health'][i % 4],
      timestamp: Date.now() - i * 60000,
      score: (i * 17) % 100,
      isFavorite: i % 5 === 0,
    }),
  );

const noop = () => {};

describe('SearchableList', () => {
  describe('rendering', () => {
    test('renders with empty list', () => {
      render(
        <SearchableList
          items={[]}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[]}
        />,
      );
      expect(screen.getByText(/0 items/)).toBeTruthy();
    });

    test('renders items grouped by category', () => {
      const items = makeItems(8);
      render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[]}
        />,
      );
      expect(screen.getAllByText(/Work/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/Personal/).length).toBeGreaterThan(0);
    });

    test('renders stats correctly', () => {
      const items = makeItems(20);
      render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[]}
        />,
      );
      expect(screen.getByText(/20 items/)).toBeTruthy();
      expect(screen.getByText(/favorites/)).toBeTruthy();
    });

    test('displays highlighted items', () => {
      const items = makeItems(5);
      const { container } = render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[1, 3]}
        />,
      );
      expect(container).toBeTruthy();
    });
  });

  describe('search', () => {
    test('filters items by label', () => {
      const items = [
        makeItem({ id: 1, label: 'Apple', category: 'Fruit' }),
        makeItem({ id: 2, label: 'Banana', category: 'Fruit' }),
        makeItem({ id: 3, label: 'Carrot', category: 'Vegetable' }),
      ];
      render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[]}
        />,
      );
      fireEvent.change(screen.getByPlaceholderText('Search...'), {
        target: { value: 'Ban' },
      });
      expect(screen.getByText('Banana')).toBeTruthy();
    });

    test('filters items by category', () => {
      const items = [
        makeItem({ id: 1, label: 'Apple', category: 'Fruit' }),
        makeItem({ id: 2, label: 'Carrot', category: 'Vegetable' }),
      ];
      render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[]}
        />,
      );
      fireEvent.change(screen.getByPlaceholderText('Search...'), {
        target: { value: 'Veg' },
      });
      expect(screen.getByText('Carrot')).toBeTruthy();
    });
  });

  describe('interactions', () => {
    test('calls onToggleFavorite when star clicked', () => {
      const onToggle = jest.fn();
      const items = [makeItem({ id: 7, isFavorite: false })];
      render(
        <SearchableList
          items={items}
          onToggleFavorite={onToggle}
          onDelete={noop}
          highlightedIds={[]}
        />,
      );
      fireEvent.click(screen.getByText('☆'));
      expect(onToggle).toHaveBeenCalledWith(7);
    });

    test('calls onDelete when delete clicked', () => {
      const onDelete = jest.fn();
      const items = [makeItem({ id: 9 })];
      render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={onDelete}
          highlightedIds={[]}
        />,
      );
      fireEvent.click(screen.getByText('×'));
      expect(onDelete).toHaveBeenCalledWith(9);
    });

    test('favorites-only filter works', () => {
      const items = [
        makeItem({ id: 1, label: 'Fav', isFavorite: true }),
        makeItem({ id: 2, label: 'NotFav', isFavorite: false }),
      ];
      render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[]}
        />,
      );
      fireEvent.click(screen.getByLabelText(/Favorites only/));
      expect(screen.getByText('Fav')).toBeTruthy();
    });
  });

  describe('performance with large datasets', () => {
    test('renders 200 items', () => {
      const items = makeItems(200);
      const { container } = render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={[1, 50, 100, 150]}
        />,
      );
      expect(container).toBeTruthy();
    });

    test('renders 500 items', () => {
      const items = makeItems(500);
      const { container } = render(
        <SearchableList
          items={items}
          onToggleFavorite={noop}
          onDelete={noop}
          highlightedIds={Array.from({ length: 50 }, (_, i) => i * 10 + 1)}
        />,
      );
      expect(container).toBeTruthy();
    });
  });
});

describe('ListItemCard', () => {
  test('renders item label and category', () => {
    const item = makeItem({ label: 'My Item', category: 'Work' });
    render(
      <ListItemCard
        item={item}
        onToggleFavorite={noop}
        onDelete={noop}
        style={{}}
        isHighlighted={false}
      />,
    );
    expect(screen.getByText('My Item')).toBeTruthy();
    expect(screen.getByText(/Work/)).toBeTruthy();
  });

  test('shows filled star for favorites', () => {
    const item = makeItem({ isFavorite: true });
    render(
      <ListItemCard
        item={item}
        onToggleFavorite={noop}
        onDelete={noop}
        style={{}}
        isHighlighted={false}
      />,
    );
    expect(screen.getByText('★')).toBeTruthy();
  });
});
