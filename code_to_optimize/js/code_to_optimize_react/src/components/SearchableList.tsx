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
import React, { useState, memo } from 'react';

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

  // Inefficient: expensive pipeline recomputed every render
  const processedItems = items
    .filter(item => {
      if (showFavoritesOnly && !item.isFavorite) return false;
      if (query) {
        return (
          item.label.toLowerCase().includes(query.toLowerCase()) ||
          item.category.toLowerCase().includes(query.toLowerCase())
        );
      }
      return true;
    })
    .sort((a, b) => {
      if (sortBy === 'label') return a.label.localeCompare(b.label);
      if (sortBy === 'score') return b.score - a.score;
      return b.timestamp - a.timestamp;
    });

  // Inefficient: grouping computed every render
  const categoryGroups = processedItems.reduce(
    (groups, item) => {
      const group = groups[item.category] || [];
      group.push(item);
      groups[item.category] = group;
      return groups;
    },
    {} as Record<string, ListItem[]>,
  );

  // Inefficient: stats computed every render
  const stats = {
    total: processedItems.length,
    favorites: processedItems.filter(i => i.isFavorite).length,
    avgScore:
      processedItems.length > 0
        ? processedItems.reduce((sum, i) => sum + i.score, 0) / processedItems.length
        : 0,
    categories: Object.keys(categoryGroups).length,
  };

  // Inefficient: creates Set on every render
  const highlightedSet = new Set(highlightedIds);

  return (
    <div>
      {/* Inefficient: inline style */}
      <div style={{ padding: '8px', display: 'flex', gap: '8px', alignItems: 'center' }}>
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

      {/* Inefficient: inline style */}
      <div style={{ padding: '4px 8px', fontSize: '12px', color: '#666' }}>
        {stats.total} items | {stats.favorites} favorites |
        Avg score: {stats.avgScore.toFixed(1)} | {stats.categories} categories
      </div>

      {Object.entries(categoryGroups).map(([category, categoryItems]) => (
        <div key={category}>
          <h3 style={{ margin: '8px 0 4px', fontSize: '14px' }}>
            {category} ({categoryItems.length})
          </h3>
          {categoryItems.map(item => (
            <ListItemCard
              key={item.id}
              item={item}
              // Inefficient: inline function creates new reference every render
              // This defeats the React.memo on ListItemCard
              onToggleFavorite={(id) => onToggleFavorite(id)}
              onDelete={(id) => onDelete(id)}
              // Inefficient: inline object creates new reference every render
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
