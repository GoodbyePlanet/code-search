import { useState, useCallback } from "react";

interface CounterProps {
  initial: number;
}

export function useCounter(initial: number) {
  const [count, setCount] = useState(initial);
  const increment = useCallback(() => setCount((c) => c + 1), []);
  return { count, increment };
}

export function Counter({ initial }: CounterProps) {
  const { count, increment } = useCounter(initial);
  return (
    <button onClick={increment}>
      Count: {count}
    </button>
  );
}
