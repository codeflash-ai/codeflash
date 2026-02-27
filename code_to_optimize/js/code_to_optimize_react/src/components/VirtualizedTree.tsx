/**
 * VirtualizedTree component - demonstrates recursive/tree data patterns.
 *
 * Intentionally inefficient patterns:
 * - Recursive tree flattening computed on every render without memoization
 * - Search/filter traverses entire tree on each keystroke
 * - Inline handler functions recreated on every render
 * - No React.memo on tree node component
 * - Path computation is O(n*depth) instead of cached
 *
 * Simulates a file tree or org chart with expandable nodes.
 */
import React, { useState } from 'react';

export interface TreeNode {
  id: string;
  label: string;
  children?: TreeNode[];
  metadata?: Record<string, string>;
  icon?: string;
}

interface FlatNode {
  node: TreeNode;
  depth: number;
  isExpanded: boolean;
  hasChildren: boolean;
  path: string[];
}

interface VirtualizedTreeProps {
  data: TreeNode[];
  onSelect: (node: TreeNode, path: string[]) => void;
  onExpand?: (nodeId: string) => void;
  onCollapse?: (nodeId: string) => void;
  selectedId?: string;
}

/**
 * Recursively flatten a tree into a list for rendering.
 * Intentionally inefficient: rebuilds entire flat list on every render.
 */
function flattenTree(
  nodes: TreeNode[],
  expandedIds: Set<string>,
  depth: number = 0,
  parentPath: string[] = [],
): FlatNode[] {
  const result: FlatNode[] = [];

  for (const node of nodes) {
    const path = [...parentPath, node.id];
    const hasChildren = !!node.children && node.children.length > 0;
    const isExpanded = expandedIds.has(node.id);

    result.push({ node, depth, isExpanded, hasChildren, path });

    if (hasChildren && isExpanded) {
      result.push(...flattenTree(node.children!, expandedIds, depth + 1, path));
    }
  }

  return result;
}

/**
 * Recursively search a tree for matching nodes.
 * Intentionally inefficient: full traversal on every keystroke.
 */
function searchTree(nodes: TreeNode[], query: string): Set<string> {
  const matchingIds = new Set<string>();
  const lowerQuery = query.toLowerCase();

  function traverse(node: TreeNode, ancestors: string[]) {
    const matches =
      node.label.toLowerCase().includes(lowerQuery) ||
      Object.values(node.metadata || {}).some(v => v.toLowerCase().includes(lowerQuery));

    if (matches) {
      matchingIds.add(node.id);
      // Also include all ancestors so the path is visible
      for (const ancestorId of ancestors) {
        matchingIds.add(ancestorId);
      }
    }

    if (node.children) {
      for (const child of node.children) {
        traverse(child, [...ancestors, node.id]);
      }
    }
  }

  for (const node of nodes) {
    traverse(node, []);
  }

  return matchingIds;
}

/**
 * Count total nodes in a tree.
 * Intentionally inefficient: full traversal every render.
 */
function countNodes(nodes: TreeNode[]): number {
  let count = 0;
  for (const node of nodes) {
    count++;
    if (node.children) {
      count += countNodes(node.children);
    }
  }
  return count;
}

/**
 * Get max depth of a tree.
 * Intentionally inefficient: full traversal every render.
 */
function getMaxDepth(nodes: TreeNode[], depth: number = 0): number {
  let maxDepth = depth;
  for (const node of nodes) {
    if (node.children && node.children.length > 0) {
      maxDepth = Math.max(maxDepth, getMaxDepth(node.children, depth + 1));
    }
  }
  return maxDepth;
}

/**
 * Tree component with search, expand/collapse, and selection.
 */
export function VirtualizedTree({
  data,
  onSelect,
  onExpand,
  onCollapse,
  selectedId,
}: VirtualizedTreeProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');

  // Inefficient: full tree traversal on every render
  const totalNodes = countNodes(data);
  const maxDepth = getMaxDepth(data);

  // Inefficient: search traversal on every render
  const matchingIds = searchQuery ? searchTree(data, searchQuery) : null;

  // Inefficient: flattening on every render
  const expandedForSearch = matchingIds
    ? new Set([...expandedIds, ...matchingIds])
    : expandedIds;
  const flatNodes = flattenTree(data, expandedForSearch);

  // Inefficient: filter on every render
  const visibleNodes = matchingIds
    ? flatNodes.filter(fn => matchingIds.has(fn.node.id))
    : flatNodes;

  // Inefficient: function recreated every render
  const toggleExpand = (nodeId: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
        onCollapse?.(nodeId);
      } else {
        next.add(nodeId);
        onExpand?.(nodeId);
      }
      return next;
    });
  };

  // Inefficient: function recreated every render
  const expandAll = () => {
    const allIds = new Set<string>();
    function collect(nodes: TreeNode[]) {
      for (const node of nodes) {
        if (node.children && node.children.length > 0) {
          allIds.add(node.id);
          collect(node.children);
        }
      }
    }
    collect(data);
    setExpandedIds(allIds);
  };

  // Inefficient: function recreated every render
  const collapseAll = () => {
    setExpandedIds(new Set());
  };

  return (
    <div>
      {/* Inefficient: inline style */}
      <div style={{ padding: '8px', borderBottom: '1px solid #e0e0e0' }}>
        <input
          type="text"
          placeholder="Search tree..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          style={{ width: '100%', padding: '4px 8px', marginBottom: '4px' }}
        />
        {/* Inefficient: inline style */}
        <div style={{ display: 'flex', gap: '8px', fontSize: '12px' }}>
          <button onClick={expandAll}>Expand All</button>
          <button onClick={collapseAll}>Collapse All</button>
          <span style={{ color: '#666' }}>
            {visibleNodes.length} of {totalNodes} nodes | Depth: {maxDepth}
          </span>
        </div>
      </div>

      <div style={{ overflow: 'auto', maxHeight: '600px' }}>
        {visibleNodes.map(flatNode => (
          <div
            key={flatNode.node.id}
            onClick={() => onSelect(flatNode.node, flatNode.path)}
            style={{
              paddingLeft: `${flatNode.depth * 20 + 8}px`,
              padding: '4px 8px',
              cursor: 'pointer',
              background: selectedId === flatNode.node.id ? '#e3f2fd' : 'transparent',
              borderBottom: '1px solid #f5f5f5',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            {flatNode.hasChildren && (
              <span
                onClick={e => {
                  e.stopPropagation();
                  toggleExpand(flatNode.node.id);
                }}
                style={{ cursor: 'pointer', width: '16px', textAlign: 'center' }}
              >
                {flatNode.isExpanded ? '▼' : '▶'}
              </span>
            )}
            {!flatNode.hasChildren && <span style={{ width: '16px' }} />}
            {flatNode.node.icon && <span>{flatNode.node.icon}</span>}
            <span>{flatNode.node.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
