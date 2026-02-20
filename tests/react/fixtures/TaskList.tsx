import React, { useState, useContext, useCallback } from 'react';

interface Task {
  id: number;
  title: string;
  completed: boolean;
  priority: 'low' | 'medium' | 'high';
}

interface TaskListProps {
  tasks: Task[];
  onToggle: (id: number) => void;
  onDelete: (id: number) => void;
  filter: 'all' | 'active' | 'completed';
}

export function TaskList({ tasks, onToggle, onDelete, filter }: TaskListProps) {
  const [sortBy, setSortBy] = useState<'title' | 'priority'>('title');

  // Inline filtering and sorting without useMemo
  const filteredTasks = tasks.filter(task => {
    if (filter === 'active') return !task.completed;
    if (filter === 'completed') return task.completed;
    return true;
  });

  const sortedTasks = filteredTasks.sort((a, b) => {
    if (sortBy === 'title') return a.title.localeCompare(b.title);
    const priority = { low: 0, medium: 1, high: 2 };
    return priority[b.priority] - priority[a.priority];
  });

  // Inline function defined in render body
  const handleToggle = (id: number) => {
    onToggle(id);
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        <button onClick={() => setSortBy('title')}>Sort by Title</button>
        <button onClick={() => setSortBy('priority')}>Sort by Priority</button>
      </div>
      <ul>
        {sortedTasks.map(task => (
          <li
            key={task.id}
            style={{
              textDecoration: task.completed ? 'line-through' : 'none',
              color: task.priority === 'high' ? 'red' : 'inherit',
            }}
          >
            <input
              type="checkbox"
              checked={task.completed}
              onChange={() => handleToggle(task.id)}
            />
            <span>{task.title}</span>
            <button onClick={() => onDelete(task.id)}>Delete</button>
          </li>
        ))}
      </ul>
      <div>Total: {sortedTasks.length} tasks</div>
    </div>
  );
}
