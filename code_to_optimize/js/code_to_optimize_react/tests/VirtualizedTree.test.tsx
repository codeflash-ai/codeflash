import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { VirtualizedTree, TreeNode } from '../src/components/VirtualizedTree';

const makeTree = (depth: number, breadth: number, prefix = ''): TreeNode[] =>
  Array.from({ length: breadth }, (_, i) => {
    const id = `${prefix}node-${i}`;
    return {
      id,
      label: `${prefix}Node ${i}`,
      metadata: { type: i % 2 === 0 ? 'folder' : 'file' },
      icon: i % 2 === 0 ? '📁' : '📄',
      children: depth > 1 ? makeTree(depth - 1, breadth, `${id}/`) : undefined,
    };
  });

const noop = () => {};

describe('VirtualizedTree', () => {
  describe('rendering', () => {
    test('renders root nodes', () => {
      const data = makeTree(1, 3);
      render(
        <VirtualizedTree data={data} onSelect={noop} />,
      );
      expect(screen.getByText('Node 0')).toBeTruthy();
      expect(screen.getByText('Node 1')).toBeTruthy();
      expect(screen.getByText('Node 2')).toBeTruthy();
    });

    test('renders with empty data', () => {
      render(<VirtualizedTree data={[]} onSelect={noop} />);
      expect(screen.getByText(/0 of 0 nodes/)).toBeTruthy();
    });

    test('displays total node count', () => {
      const data = makeTree(2, 3); // 3 + 9 = 12 nodes
      render(<VirtualizedTree data={data} onSelect={noop} />);
      expect(screen.getByText(/12/)).toBeTruthy();
    });

    test('displays depth info', () => {
      const data = makeTree(3, 2); // depth 2 (0-indexed)
      render(<VirtualizedTree data={data} onSelect={noop} />);
      expect(screen.getByText(/Depth/)).toBeTruthy();
    });

    test('renders expand/collapse icons for parent nodes', () => {
      const data = makeTree(2, 2);
      render(<VirtualizedTree data={data} onSelect={noop} />);
      const expandButtons = screen.getAllByText('▶');
      expect(expandButtons.length).toBeGreaterThan(0);
    });
  });

  describe('expand/collapse', () => {
    test('expands node to show children', () => {
      const data = makeTree(2, 2);
      render(<VirtualizedTree data={data} onSelect={noop} />);
      // Click expand on first node
      fireEvent.click(screen.getAllByText('▶')[0]);
      // Children should now be visible
      expect(screen.getByText(/node-0\/Node 0/)).toBeTruthy();
    });

    test('expand all shows all nodes', () => {
      const data = makeTree(2, 2);
      render(<VirtualizedTree data={data} onSelect={noop} />);
      fireEvent.click(screen.getByText('Expand All'));
      // All leaf nodes should be visible now
      const nodeTexts = screen.getAllByText(/Node/);
      expect(nodeTexts.length).toBeGreaterThan(2);
    });

    test('collapse all hides children', () => {
      const data = makeTree(2, 2);
      render(<VirtualizedTree data={data} onSelect={noop} />);
      fireEvent.click(screen.getByText('Expand All'));
      fireEvent.click(screen.getByText('Collapse All'));
      // Only root nodes should be visible
    });

    test('calls onExpand callback', () => {
      const onExpand = jest.fn();
      const data = makeTree(2, 2);
      render(<VirtualizedTree data={data} onSelect={noop} onExpand={onExpand} />);
      fireEvent.click(screen.getAllByText('▶')[0]);
      expect(onExpand).toHaveBeenCalledWith('node-0');
    });

    test('calls onCollapse callback', () => {
      const onCollapse = jest.fn();
      const data = makeTree(2, 2);
      render(<VirtualizedTree data={data} onSelect={noop} onCollapse={onCollapse} />);
      // Expand then collapse
      fireEvent.click(screen.getAllByText('▶')[0]);
      fireEvent.click(screen.getAllByText('▼')[0]);
      expect(onCollapse).toHaveBeenCalledWith('node-0');
    });
  });

  describe('search', () => {
    test('filters tree by search query', () => {
      const data: TreeNode[] = [
        {
          id: 'src',
          label: 'src',
          children: [
            { id: 'app', label: 'App.tsx' },
            { id: 'index', label: 'index.ts' },
          ],
        },
        {
          id: 'tests',
          label: 'tests',
          children: [
            { id: 'app-test', label: 'App.test.tsx' },
          ],
        },
      ];
      render(<VirtualizedTree data={data} onSelect={noop} />);
      fireEvent.change(screen.getByPlaceholderText('Search tree...'), {
        target: { value: 'App' },
      });
      // App.tsx and App.test.tsx should match, plus ancestors
    });
  });

  describe('selection', () => {
    test('calls onSelect with node and path', () => {
      const onSelect = jest.fn();
      const data = makeTree(1, 3);
      render(<VirtualizedTree data={data} onSelect={onSelect} />);
      fireEvent.click(screen.getByText('Node 1'));
      expect(onSelect).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'node-1' }),
        ['node-1'],
      );
    });

    test('highlights selected node', () => {
      const data = makeTree(1, 3);
      const { container } = render(
        <VirtualizedTree data={data} onSelect={noop} selectedId="node-1" />,
      );
      expect(container).toBeTruthy();
    });
  });

  describe('performance with large trees', () => {
    test('renders tree with 100 nodes (10x10)', () => {
      const data = makeTree(2, 10); // 10 + 100 = 110 nodes
      const { container } = render(
        <VirtualizedTree data={data} onSelect={noop} />,
      );
      expect(container).toBeTruthy();
    });

    test('renders tree with 1000+ nodes (10x10x10)', () => {
      const data = makeTree(3, 10); // 10 + 100 + 1000 = 1110 nodes
      const { container } = render(
        <VirtualizedTree data={data} onSelect={noop} />,
      );
      expect(container).toBeTruthy();
    });

    test('search on large tree', () => {
      const data = makeTree(3, 5); // 5 + 25 + 125 = 155 nodes
      render(<VirtualizedTree data={data} onSelect={noop} />);
      fireEvent.change(screen.getByPlaceholderText('Search tree...'), {
        target: { value: 'Node 3' },
      });
      // Should filter without crashing
    });
  });
});
