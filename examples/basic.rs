use num_complex::Complex;
use std::f64::consts::PI;
use qm::test;
use ndarray::prelude::*;
// use ndarray_linalg::trace::*;

pub type ComplexNum = num_complex::Complex<f64>;
pub type Vec = ndarray::Array1<ComplexNum>;
pub type SqrMatrix = ndarray::Array2<ComplexNum>;

fn main() {
  let _message = test();
  println!("The 20th Fibonacci number is {}", fib(20));
  println!("Fib(2.1 -3.3i) is {}", fibc(Complex{re:2.1, im:-3.3}));

  let b: Vec = array![Complex{re:2., im:0.},Complex{re:1., im:0.}];
  let a: Vec = array![Complex{re:3., im:0.},Complex{re:4., im:0.}];

  println!("{}", a.dot(&b));
  println!("{}",density_matrix(Complex{re:1., im:2.},Complex{re:3., im:4.}))
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
 let ln_phi: ComplexNum = Complex::<f64>::from(phi.ln());
 let phi_to_the_n: ComplexNum = (n*ln_phi).exp();
 let phi_to_the_neg_n: ComplexNum = (-n*ln_phi).exp();
 (phi_to_the_n - ((PI*n).cos())*phi_to_the_neg_n)/5_f64.sqrt()
}

pub fn density_matrix(alpha: ComplexNum, beta: ComplexNum) -> SqrMatrix{

 // let zero_ket: Vec = array![Complex{re:1., im:0.},Complex{re:0., im:0.}];
 // let one_ket: Vec = array![Complex{re:0., im:0.},Complex{re:1., im:0.}];
 // let zero_ket_conj: Vec = array![zero_ket[0].conj(),zero_ket[1].conj()];
 // let one_ket_conj: Vec = array![zero_ket[0].conj(),zero_ket[1].conj()];
 let rho00: ComplexNum = alpha*alpha.conj();
 let rho01: ComplexNum = alpha*beta.conj();
 let rho10: ComplexNum = beta*alpha.conj(); 
 let rho11: ComplexNum = beta*beta.conj();
 array![[rho00,rho01],[rho10,rho11]]
}

pub fn find_purity(rho_squared: Array2<f64>)-> f64{
  let diag = rho_squared.diag();
  diag.sum()
  // rho_squared.trace()
}