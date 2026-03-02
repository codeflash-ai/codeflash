import React, { memo } from 'react';

interface ListItemProps {
  text: string;
  isSelected: boolean;
}

const ListItem = memo(function ListItem({ text, isSelected }: ListItemProps) {
  return (
    <li className={isSelected ? 'selected' : ''}>
      {text}
    </li>
  );
});

interface MemoizedListProps {
  items: string[];
  selectedIndex: number;
}

export const MemoizedList = memo(function MemoizedList({ items, selectedIndex }: MemoizedListProps) {
  return (
    <ul>
      {items.map((item, index) => (
        <ListItem key={index} text={item} isSelected={index === selectedIndex} />
      ))}
    </ul>
  );
});
