/**
 * useDebounce hook - standard debounce implementation.
 *
 * This hook is already reasonably implemented. It should be detected as a hook
 * (not a component) and handled appropriately by the optimization pipeline.
 */
import { useState, useEffect } from 'react';

/**
 * Debounce a value by the given delay in milliseconds.
 * Returns the debounced value that only updates after the delay.
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}
