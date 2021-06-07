use num_complex::Complex;
use std::f64::consts::PI;
use qm::test;

fn main() {
  let _message = test();
  println!("The 20th Fibonacci number is {}", fib(20));
  println!("n is {}", fibc(Complex{re:2.2, im:0.0}));
}

pub fn fib(n : u64) -> u64 {
 
 let n_int = n;
 let n_float = n_int as f64;
 let phi: f64 = (1_f64 + 5_f64.sqrt())/2_f64;
 let fib_num: f64 = ((phi.powf(n_float)/5_f64.sqrt()) + 0.5).floor();
 fib_num as u64
}


pub fn fibc(n: Complex<f64>) -> Complex<f64>{
 
 let phi: f64 = (1.0 + 5_f64.sqrt())/2.0;
 let phi_to_the_n: Complex<f64> = (n*Complex::<f64>::from(phi.ln())).exp();
 let phi_to_the_neg_n: Complex<f64> = (-n*Complex::<f64>::from(phi.ln())).exp();
 (phi_to_the_n - ((PI*n).cos())*phi_to_the_neg_n)/5_f64.sqrt()
}
