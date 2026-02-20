/**
 * TaskBoard component - demonstrates multiple React optimization opportunities.
 *
 * Intentionally inefficient patterns:
 * - Inline object/array literals in JSX props (new reference every render)
 * - Functions defined in render body without useCallback
 * - Expensive .filter/.sort/.reduce without useMemo
 * - Component not wrapped in React.memo
 *
 * These patterns cause unnecessary re-renders in child components when TaskBoard's
 * parent re-renders, even if TaskBoard's props haven't changed.
 */
import React, { useState } from 'react';

export interface Task {
  id: number;
  title: string;
  description: string;
  status: 'todo' | 'in-progress' | 'done';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignee: string;
  tags: string[];
  createdAt: number;
  updatedAt: number;
}

interface TaskBoardProps {
  tasks: Task[];
  onTaskUpdate: (id: number, updates: Partial<Task>) => void;
  onTaskDelete: (id: number) => void;
  onTaskCreate: (task: Omit<Task, 'id' | 'createdAt' | 'updatedAt'>) => void;
  currentUser: string;
}

const PRIORITY_ORDER = { critical: 0, high: 1, medium: 2, low: 3 };

/**
 * Main task board component with filtering, sorting, and grouping.
 * Contains multiple optimization opportunities for codeflash to detect and fix.
 */
export function TaskBoard({
  tasks,
  onTaskUpdate,
  onTaskDelete,
  onTaskCreate,
  currentUser,
}: TaskBoardProps) {
  const [filterStatus, setFilterStatus] = useState<Task['status'] | 'all'>('all');
  const [filterAssignee, setFilterAssignee] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'priority' | 'title' | 'createdAt'>('priority');
  const [searchQuery, setSearchQuery] = useState('');

  // Inefficient: runs on every render, should use useMemo
  const filteredTasks = tasks.filter(task => {
    if (filterStatus !== 'all' && task.status !== filterStatus) return false;
    if (filterAssignee !== 'all' && task.assignee !== filterAssignee) return false;
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        task.title.toLowerCase().includes(query) ||
        task.description.toLowerCase().includes(query) ||
        task.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }
    return true;
  });

  // Inefficient: runs on every render, should use useMemo
  const sortedTasks = filteredTasks.sort((a, b) => {
    if (sortBy === 'priority') {
      return PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority];
    }
    if (sortBy === 'title') {
      return a.title.localeCompare(b.title);
    }
    return b.createdAt - a.createdAt;
  });

  // Inefficient: expensive grouping runs on every render, should use useMemo
  const groupedTasks = sortedTasks.reduce(
    (groups, task) => {
      const group = groups[task.status] || [];
      group.push(task);
      groups[task.status] = group;
      return groups;
    },
    {} as Record<Task['status'], Task[]>,
  );

  // Inefficient: computed on every render, should use useMemo
  const uniqueAssignees = Array.from(new Set(tasks.map(t => t.assignee))).sort();

  // Inefficient: computed on every render, should use useMemo
  const stats = {
    total: tasks.length,
    todo: tasks.filter(t => t.status === 'todo').length,
    inProgress: tasks.filter(t => t.status === 'in-progress').length,
    done: tasks.filter(t => t.status === 'done').length,
    myTasks: tasks.filter(t => t.assignee === currentUser).length,
    critical: tasks.filter(t => t.priority === 'critical' && t.status !== 'done').length,
  };

  // Inefficient: function created on every render, should use useCallback
  const handleStatusChange = (taskId: number, newStatus: Task['status']) => {
    onTaskUpdate(taskId, { status: newStatus, updatedAt: Date.now() });
  };

  // Inefficient: function created on every render, should use useCallback
  const handlePriorityChange = (taskId: number, newPriority: Task['priority']) => {
    onTaskUpdate(taskId, { priority: newPriority, updatedAt: Date.now() });
  };

  // Inefficient: function created on every render, should use useCallback
  const handleDelete = (taskId: number) => {
    onTaskDelete(taskId);
  };

  // Inefficient: function created on every render, should use useCallback
  const handleQuickCreate = () => {
    onTaskCreate({
      title: 'New Task',
      description: '',
      status: 'todo',
      priority: 'medium',
      assignee: currentUser,
      tags: [],
    });
  };

  return (
    <div>
      {/* Inefficient: inline style object creates new reference every render */}
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px' }}>
        <h1>Task Board</h1>
        <button onClick={handleQuickCreate}>+ New Task</button>
      </div>

      {/* Inefficient: inline style object */}
      <div style={{ display: 'flex', gap: '8px', padding: '0 16px', flexWrap: 'wrap' }}>
        <span>Status: {stats.todo} todo, {stats.inProgress} in progress, {stats.done} done</span>
        <span>| My tasks: {stats.myTasks}</span>
        {stats.critical > 0 && (
          <span style={{ color: 'red', fontWeight: 'bold' }}>
            {stats.critical} critical!
          </span>
        )}
      </div>

      {/* Inefficient: inline style object */}
      <div style={{ display: 'flex', gap: '8px', padding: '16px' }}>
        <input
          type="text"
          placeholder="Search tasks..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
        <select value={filterStatus} onChange={e => setFilterStatus(e.target.value as Task['status'] | 'all')}>
          <option value="all">All Statuses</option>
          <option value="todo">To Do</option>
          <option value="in-progress">In Progress</option>
          <option value="done">Done</option>
        </select>
        <select value={filterAssignee} onChange={e => setFilterAssignee(e.target.value)}>
          <option value="all">All Assignees</option>
          {uniqueAssignees.map(assignee => (
            <option key={assignee} value={assignee}>{assignee}</option>
          ))}
        </select>
        <select value={sortBy} onChange={e => setSortBy(e.target.value as typeof sortBy)}>
          <option value="priority">Sort by Priority</option>
          <option value="title">Sort by Title</option>
          <option value="createdAt">Sort by Date</option>
        </select>
      </div>

      {/* Inefficient: inline style object */}
      <div style={{ display: 'flex', gap: '16px', padding: '16px' }}>
        {(['todo', 'in-progress', 'done'] as Task['status'][]).map(status => (
          <div key={status} style={{ flex: 1, minWidth: '250px' }}>
            <h2>{status} ({(groupedTasks[status] || []).length})</h2>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {(groupedTasks[status] || []).map(task => (
                <li
                  key={task.id}
                  style={{
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    padding: '8px',
                    marginBottom: '8px',
                    borderLeft: `4px solid ${
                      task.priority === 'critical' ? '#d32f2f'
                        : task.priority === 'high' ? '#f57c00'
                          : task.priority === 'medium' ? '#1976d2'
                            : '#388e3c'
                    }`,
                  }}
                >
                  <strong>{task.title}</strong>
                  <p style={{ margin: '4px 0', fontSize: '12px', color: '#666' }}>
                    {task.description}
                  </p>
                  {/* Inefficient: inline style object */}
                  <div style={{ display: 'flex', gap: '4px', fontSize: '11px' }}>
                    {task.tags.map(tag => (
                      <span
                        key={tag}
                        style={{
                          background: '#e0e0e0',
                          borderRadius: '2px',
                          padding: '1px 4px',
                        }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  {/* Inefficient: inline style object */}
                  <div style={{ display: 'flex', gap: '4px', marginTop: '8px' }}>
                    <select
                      value={task.status}
                      onChange={e => handleStatusChange(task.id, e.target.value as Task['status'])}
                    >
                      <option value="todo">To Do</option>
                      <option value="in-progress">In Progress</option>
                      <option value="done">Done</option>
                    </select>
                    <select
                      value={task.priority}
                      onChange={e => handlePriorityChange(task.id, e.target.value as Task['priority'])}
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                      <option value="critical">Critical</option>
                    </select>
                    <button
                      onClick={() => handleDelete(task.id)}
                      style={{ color: 'red', border: 'none', cursor: 'pointer' }}
                    >
                      Delete
                    </button>
                  </div>
                  <small style={{ color: '#999' }}>
                    Assigned to: {task.assignee}
                  </small>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
