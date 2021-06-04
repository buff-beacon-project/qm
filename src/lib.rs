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

pub fn fib_practice(n : u64) -> u64 {
 
 let n_int = n;
 let n_float = n_int as f64;
 let phi: f64 = (1_f64 + 5_f64.sqrt())/2_f64;
 let fib_num: f64 = ((phi.powf(n_float)/5_f64.sqrt()) + 0.5).floor();
 fib_num as u64
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_fib() {
    let actual = fib_practice(19);
    let expected = 4181;
    assert_eq!(expected, actual);
  }
}

