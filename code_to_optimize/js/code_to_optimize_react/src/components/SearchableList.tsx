/**
 * SearchableList component - demonstrates referential instability patterns.
 *
 * Intentionally inefficient patterns:
 * - Inline function props passed to memoized children (breaks memo)
 * - Inline object props creating new references on every render
 * - Derived data (search, sort, group) recomputed without memoization
 * - Component not wrapped in React.memo
 *
 * This is a common pattern in codebases where developers wrap children in memo()
 * but forget to stabilize the parent's prop references, negating the benefit.
 */
import React, { useState, memo , useMemo } from 'react';

const H3_STYLE = { margin: '8px 0 4px', fontSize: '14px' } as const;

const STATS_STYLE = { padding: '4px 8px', fontSize: '12px', color: '#666' } as const;

const CONTAINER_STYLE = { padding: '8px', display: 'flex', gap: '8px', alignItems: 'center' } as const;

export interface ListItem {
  id: number;
  label: string;
  category: string;
  timestamp: number;
  score: number;
  isFavorite: boolean;
}

interface ListItemCardProps {
  item: ListItem;
  onToggleFavorite: (id: number) => void;
  onDelete: (id: number) => void;
  style: React.CSSProperties;
  isHighlighted: boolean;
}

/**
 * Memoized child component — but parent passes unstable references,
 * so React.memo never actually prevents re-renders.
 */
export const ListItemCard = memo(function ListItemCard({
  item,
  onToggleFavorite,
  onDelete,
  style,
  isHighlighted,
}: ListItemCardProps) {
  return (
    <div style={style}>
      <span style={{ fontWeight: isHighlighted ? 'bold' : 'normal' }}>
        {item.label}
      </span>
      <span> [{item.category}]</span>
      <span> Score: {item.score}</span>
      <button onClick={() => onToggleFavorite(item.id)}>
        {item.isFavorite ? '★' : '☆'}
      </button>
      <button onClick={() => onDelete(item.id)}>×</button>
    </div>
  );
});

interface SearchableListProps {
  items: ListItem[];
  onToggleFavorite: (id: number) => void;
  onDelete: (id: number) => void;
  highlightedIds: number[];
}

/**
 * Parent component that defeats child memoization by passing unstable references.
 */
export function SearchableList({
  items,
  onToggleFavorite,
  onDelete,
  highlightedIds,
}: SearchableListProps) {
  const [query, setQuery] = useState('');
  const [sortBy, setSortBy] = useState<'label' | 'score' | 'timestamp'>('label');
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);

  // Optimized: memoize filtering, sorting, grouping and stats into a single useMemo to avoid repeated array scans
  const { processedItems, categoryGroups, stats } = useMemo(() => {
    const q = query ? query.toLowerCase() : '';
    // Filter into a new array in one pass
    const filtered: ListItem[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (showFavoritesOnly && !item.isFavorite) continue;
      if (q) {
        const label = item.label.toLowerCase();
        const category = item.category.toLowerCase();
        if (!label.includes(q) && !category.includes(q)) continue;
      }
      filtered.push(item);
    }

    // Sort the filtered array
    if (sortBy === 'label') {
      filtered.sort((a, b) => a.label.localeCompare(b.label));
    } else if (sortBy === 'score') {
      filtered.sort((a, b) => b.score - a.score);
    } else {
      filtered.sort((a, b) => b.timestamp - a.timestamp);
    }

    // Build groups and aggregate stats in a single pass over the sorted filtered array
    const groups: Record<string, ListItem[]> = {};
    let favorites = 0;
    let sumScore = 0;
    for (let i = 0; i < filtered.length; i++) {
      const it = filtered[i];
      const g = groups[it.category];
      if (g) g.push(it);
      else groups[it.category] = [it];
      if (it.isFavorite) favorites++;
      sumScore += it.score;
    }

    const total = filtered.length;
    const avgScore = total > 0 ? sumScore / total : 0;
    const categories = Object.keys(groups).length;

    return {
      processedItems: filtered,
      categoryGroups: groups,
      stats: { total, favorites, avgScore, categories },
    };
  }, [items, showFavoritesOnly, query, sortBy]);

  // Optimized: create Set once per highlightedIds change
  const highlightedSet = useMemo(() => new Set(highlightedIds), [highlightedIds]);

  return (
    <div>
      <div style={CONTAINER_STYLE}>
        <input
          type="text"
          placeholder="Search..."
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <select value={sortBy} onChange={e => setSortBy(e.target.value as typeof sortBy)}>
          <option value="label">By Name</option>
          <option value="score">By Score</option>
          <option value="timestamp">By Date</option>
        </select>
        <label>
          <input
            type="checkbox"
            checked={showFavoritesOnly}
            onChange={e => setShowFavoritesOnly(e.target.checked)}
          />
          Favorites only
        </label>
      </div>

      <div style={STATS_STYLE}>
        {stats.total} items | {stats.favorites} favorites |
        Avg score: {stats.avgScore.toFixed(1)} | {stats.categories} categories
      </div>

      {Object.entries(categoryGroups).map(([category, categoryItems]) => (
        <div key={category}>
          <h3 style={H3_STYLE}>
            {category} ({categoryItems.length})
          </h3>
          {categoryItems.map(item => (
            <ListItemCard
              key={item.id}
              item={item}
              onToggleFavorite={onToggleFavorite}
              onDelete={onDelete}
              style={{
                padding: '4px 8px',
                borderBottom: '1px solid #eee',
                background: highlightedSet.has(item.id) ? '#fff3e0' : 'white',
              }}
              isHighlighted={highlightedSet.has(item.id)}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
