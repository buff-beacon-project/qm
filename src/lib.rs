use num_complex::Complex;
use std::f64::consts::PI;

pub fn test() -> String {
  "Hello world".into()
}

/// Get the "n"th fibonacci number using integers only.
pub fn fib(n : u64) -> u64 {
 
 let n_int = n;
 let n_float = n_int as f64;
 let phi: f64 = (1_f64 + 5_f64.sqrt())/2_f64;
 let fib_num: f64 = ((phi.powf(n_float)/5_f64.sqrt()) + 0.5).floor();
 fib_num as u64
}

/// Get the "n"th fibonacci number using integers only.
pub fn fibc(n: Complex<f64>) -> Complex<f64>{
 
 let phi: f64 = (1.0 + 5_f64.sqrt())/2.0;
 let phi_to_the_n: Complex<f64> = (n*Complex::<f64>::from(phi.ln())).exp();
 let phi_to_the_neg_n: Complex<f64> = (-n*Complex::<f64>::from(phi.ln())).exp();
 (phi_to_the_n - ((PI*n).cos())*phi_to_the_neg_n)/5_f64.sqrt()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_fibc() {
    let actual = fibc(Complex{re:1.1, im:-4.0});
    let expected = Complex{re:-23399.5652802858096777, im: 29643.22961747001006496};
    assert!((actual - expected).norm_sqr() <= 1e-6);
  }
}

