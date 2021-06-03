pub fn test() -> String {
  "Hello world".into()
}

/// Get the "n"th fibonacci number
pub fn fib(n : u64) -> u64 {
  if n == 0 {
    0
  } else if n == 1 {
    1
  } else {
    fib(n - 1) + fib(n - 2)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_fib() {
    let actual = fib(18);
    let expected = 2584;
    assert_eq!(expected, actual);
  }
}

