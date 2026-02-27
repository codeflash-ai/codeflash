/**
 * OptimizedCounter component - already properly optimized.
 *
 * This component correctly uses:
 * - React.memo wrapping
 * - useCallback for handlers
 * - useMemo for computed values
 *
 * The optimizer should detect this is already optimized and skip it
 * (or produce minimal/no changes).
 */
import React, { useState, useCallback, useMemo, memo } from 'react';

interface CounterProps {
  initialCount: number;
  step: number;
  label: string;
  max?: number;
  min?: number;
}

export const OptimizedCounter = memo(function OptimizedCounter({
  initialCount,
  step,
  label,
  max = Infinity,
  min = -Infinity,
}: CounterProps) {
  const [count, setCount] = useState(initialCount);

  const increment = useCallback(() => {
    setCount(c => Math.min(c + step, max));
  }, [step, max]);

  const decrement = useCallback(() => {
    setCount(c => Math.max(c - step, min));
  }, [step, min]);

  const reset = useCallback(() => {
    setCount(initialCount);
  }, [initialCount]);

  const displayValue = useMemo(() => {
    if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
    if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
    return String(count);
  }, [count]);

  const isAtMax = useMemo(() => count >= max, [count, max]);
  const isAtMin = useMemo(() => count <= min, [count, min]);

  return (
    <div>
      <span>{label}: {displayValue}</span>
      <button onClick={decrement} disabled={isAtMin}>-{step}</button>
      <button onClick={increment} disabled={isAtMax}>+{step}</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
});
