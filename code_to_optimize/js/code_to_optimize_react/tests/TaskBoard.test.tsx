import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { TaskBoard, Task } from '../src/components/TaskBoard';

const createTask = (overrides: Partial<Task> = {}): Task => ({
  id: 1,
  title: 'Test Task',
  description: 'A test task description',
  status: 'todo',
  priority: 'medium',
  assignee: 'alice',
  tags: ['frontend'],
  createdAt: Date.now(),
  updatedAt: Date.now(),
  ...overrides,
});

const makeTasks = (count: number): Task[] =>
  Array.from({ length: count }, (_, i) =>
    createTask({
      id: i + 1,
      title: `Task ${i + 1}`,
      status: (['todo', 'in-progress', 'done'] as const)[i % 3],
      priority: (['low', 'medium', 'high', 'critical'] as const)[i % 4],
      assignee: ['alice', 'bob', 'charlie'][i % 3],
      tags: [`tag-${i % 5}`, `category-${i % 3}`],
      createdAt: Date.now() - i * 100000,
    }),
  );

const noop = () => {};

describe('TaskBoard', () => {
  describe('rendering', () => {
    test('renders with empty task list', () => {
      render(
        <TaskBoard
          tasks={[]}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      expect(screen.getByText('Task Board')).toBeTruthy();
    });

    test('renders all status columns', () => {
      const tasks = makeTasks(9);
      render(
        <TaskBoard
          tasks={tasks}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      expect(screen.getByText(/todo/i)).toBeTruthy();
      expect(screen.getByText(/in-progress/i)).toBeTruthy();
      expect(screen.getByText(/done/i)).toBeTruthy();
    });

    test('renders task titles', () => {
      const tasks = [createTask({ id: 1, title: 'Important Feature' })];
      render(
        <TaskBoard
          tasks={tasks}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      expect(screen.getByText('Important Feature')).toBeTruthy();
    });

    test('renders stats correctly', () => {
      const tasks = [
        createTask({ id: 1, status: 'todo' }),
        createTask({ id: 2, status: 'todo' }),
        createTask({ id: 3, status: 'in-progress' }),
        createTask({ id: 4, status: 'done' }),
      ];
      render(
        <TaskBoard
          tasks={tasks}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      expect(screen.getByText(/2 todo/)).toBeTruthy();
      expect(screen.getByText(/1 in progress/)).toBeTruthy();
      expect(screen.getByText(/1 done/)).toBeTruthy();
    });
  });

  describe('filtering', () => {
    test('search filters tasks by title', () => {
      const tasks = [
        createTask({ id: 1, title: 'Build UI' }),
        createTask({ id: 2, title: 'Fix API bug' }),
        createTask({ id: 3, title: 'Write docs' }),
      ];
      render(
        <TaskBoard
          tasks={tasks}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      const searchInput = screen.getByPlaceholderText('Search tasks...');
      fireEvent.change(searchInput, { target: { value: 'API' } });
      expect(screen.getByText('Fix API bug')).toBeTruthy();
    });
  });

  describe('interactions', () => {
    test('calls onTaskDelete when delete button clicked', () => {
      const onDelete = jest.fn();
      const tasks = [createTask({ id: 42, title: 'Delete me' })];
      render(
        <TaskBoard
          tasks={tasks}
          onTaskUpdate={noop}
          onTaskDelete={onDelete}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      fireEvent.click(screen.getByText('Delete'));
      expect(onDelete).toHaveBeenCalledWith(42);
    });

    test('calls onTaskCreate when new task button clicked', () => {
      const onCreate = jest.fn();
      render(
        <TaskBoard
          tasks={[]}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={onCreate}
          currentUser="alice"
        />,
      );
      fireEvent.click(screen.getByText('+ New Task'));
      expect(onCreate).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'New Task', assignee: 'alice' }),
      );
    });
  });

  describe('performance with large datasets', () => {
    test('renders 100 tasks without error', () => {
      const tasks = makeTasks(100);
      const { container } = render(
        <TaskBoard
          tasks={tasks}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      expect(container).toBeTruthy();
    });

    test('renders 500 tasks without error', () => {
      const tasks = makeTasks(500);
      const { container } = render(
        <TaskBoard
          tasks={tasks}
          onTaskUpdate={noop}
          onTaskDelete={noop}
          onTaskCreate={noop}
          currentUser="alice"
        />,
      );
      expect(container).toBeTruthy();
    });
  });
});
