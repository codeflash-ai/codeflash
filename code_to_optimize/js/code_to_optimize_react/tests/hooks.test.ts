import { renderHook, act } from '@testing-library/react';
import { useDebounce } from '../src/hooks/useDebounce';

describe('useDebounce', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('returns initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('hello', 500));
    expect(result.current).toBe('hello');
  });

  test('does not update value before delay', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 'hello', delay: 500 } },
    );

    rerender({ value: 'world', delay: 500 });
    expect(result.current).toBe('hello');
  });

  test('updates value after delay', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 'hello', delay: 500 } },
    );

    rerender({ value: 'world', delay: 500 });
    act(() => {
      jest.advanceTimersByTime(500);
    });
    expect(result.current).toBe('world');
  });

  test('resets timer on rapid changes', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 'a', delay: 300 } },
    );

    rerender({ value: 'ab', delay: 300 });
    act(() => { jest.advanceTimersByTime(100); });

    rerender({ value: 'abc', delay: 300 });
    act(() => { jest.advanceTimersByTime(100); });

    rerender({ value: 'abcd', delay: 300 });
    act(() => { jest.advanceTimersByTime(100); });

    // Not yet debounced
    expect(result.current).toBe('a');

    act(() => { jest.advanceTimersByTime(300); });
    expect(result.current).toBe('abcd');
  });

  test('works with numbers', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 0, delay: 200 } },
    );

    rerender({ value: 42, delay: 200 });
    act(() => { jest.advanceTimersByTime(200); });
    expect(result.current).toBe(42);
  });

  test('works with objects', () => {
    const obj1 = { name: 'Alice' };
    const obj2 = { name: 'Bob' };
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: obj1, delay: 100 } },
    );

    rerender({ value: obj2, delay: 100 });
    act(() => { jest.advanceTimersByTime(100); });
    expect(result.current).toEqual({ name: 'Bob' });
  });
});
