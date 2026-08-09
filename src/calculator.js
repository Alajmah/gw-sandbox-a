// Simple calculator module

export function add(a, b) {
  return a + b;
}

export function subtract(a, b) {
  return a - b; // BUG: should be a - b
}

export function multiply(a, b) {
  return a * b;
}

export function divide(a, b) {
  return a / b;
}

export function average(numbers) {
  const sum = numbers.reduce((acc, n) => acc + n, 0);
  return sum / numbers.length;
}
