import React, { useState } from 'react';

interface CounterProps {
  initialCount?: number;
  label?: string;
}

export function Counter({ initialCount = 0, label = 'Count' }: CounterProps) {
  const [count, setCount] = useState(initialCount);

  const increment = () => setCount(c => c + 1);
  const decrement = () => setCount(c => c - 1);

  return (
    <div>
      <span>{label}: {count}</span>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
    </div>
  );
}
