use std::f64::consts::PI;
use std::vec::*;
use std::cmp;
use ndarray::prelude::*;
use ndarray_linalg::*;

pub type Complex = num_complex::Complex<f64>;
pub type Vec = ndarray::Array1<c64>;
pub type MatrixC64 = ndarray::Array2<c64>;
pub type MatrixF64 = ndarray::Array2<f64>;

/// Get the "n"th fibonacci number using integers only.
pub fn fib(n : u64) -> u64 {
 
 let n_float = n as f64;
 let phi = ( 1. + 5_f64.sqrt() ) / 2.;
 let fib_num = ( (phi.powf(n_float) / 5_f64.sqrt() ) + 0.5 ).floor();

 fib_num as u64

}

/// Get the "n"th fibonacci number using complex numbers. For example, n = Complex::new(2.1, -3.3).
pub fn fibc(n: Complex) -> Complex{
 
 let phi: f64 = ( 1. + 5_f64.sqrt() ) / 2.;
 let ln_phi = Complex::from( phi.ln() );
 let phi_to_the_n = (n * ln_phi).exp();
 let phi_to_the_neg_n = (-n * ln_phi).exp();

 (phi_to_the_n - ( (PI * n).cos() ) * phi_to_the_neg_n) / 5_f64.sqrt()

}

pub fn create_dens_matrix(coefs: Vec) -> MatrixC64{

  let coef_num = coefs.len() as i32;
  let dens_mat_len = coef_num as usize;
  let mut dens_matrix = MatrixC64::zeros((dens_mat_len,dens_mat_len).f());

  let mut ctr_i = 0;

  for _coef in 0..coef_num{
    let mut ctr_j = 0;
    for _coef_2 in 0..coef_num{
      dens_matrix[ [ctr_i , ctr_j] ] = coefs[ctr_i]*(coefs[ctr_j]).conj();
      ctr_j += 1;
    }
  ctr_i +=1;
  }
  
  dens_matrix
}


pub fn find_dens_matrix_sqrd(rho: MatrixC64) -> MatrixC64{
  let rho_sqrd = rho.dot(&rho); 
  rho_sqrd
}

pub fn find_purity(rho_sqrd: MatrixC64)-> f64{
  let purity = rho_sqrd.trace().unwrap();
  purity.re
}
pub fn find_dim(rho_sqrd: MatrixC64)-> f64{
  let shape = rho_sqrd.dim();
  shape.1 as f64
}

pub fn fidelity(rho: MatrixC64, sigma: MatrixC64) -> f64 {

  let sqrt_rho = rho.ssqrt(UPLO::Lower).unwrap();
  let product = sqrt_rho.dot(&sigma).dot(&sqrt_rho);
  let sqrt_product = product.ssqrt(UPLO::Lower).unwrap();
  (sqrt_product.trace().unwrap()).re
}

pub fn find_concurrence(rho: MatrixC64) -> f64{

  let pauli_y = array![ [  c64::new(0. , 0.)  ,  c64::new(0. , 0.) , c64::new(0. , 0.) ,  c64::new(-1. , 0.)  ] , 
                        [  c64::new(0. , 0.)  ,  c64::new(0. , 0.) , c64::new(1. , 0.) ,  c64::new(0. , 0.)   ] ,
                        [  c64::new(0. , 0.)  ,  c64::new(1. , 0.) , c64::new(0. , 0.) ,  c64::new(0. , 0.)   ] ,
                        [  c64::new(-1. , 0.) ,  c64::new(0. , 0.) , c64::new(0. , 0.) ,  c64::new(0. , 0.)   ] ];
  println!("rho is {}", rho);
  let sqrt_rho_lower = rho.ssqrt(UPLO::Upper);
  println!("sqrt_root_lower = \n {:?}\n ",sqrt_rho_lower);
  let sqrt_rho = sqrt_rho_lower.unwrap();

  let rho_star = rho.mapv(|rho| rho.conj());
  let rho_tilde = pauli_y.dot(&rho_star).dot(&pauli_y);

  let product = sqrt_rho.dot(&rho_tilde).dot(&sqrt_rho);
  
  let sqrt_product = product.ssqrt(UPLO::Lower).unwrap();

  let (eigvals, _eigvecs) = sqrt_product.eigh(UPLO::Lower).unwrap();
  
  println!("eigs = {}",eigvals);
  0_f64.max(eigvals[3] - eigvals[2] - eigvals[1] - eigvals[0])
}

// pub fn find_sqr_root_of_matrix(rho: MatrixC64) -> MatrixC64 {

// //Hermitian matrices are always diagonalizable, so we can express rho = S*D*(S^-1),
// //so the sqrt(rho) = S*(D^0.5)*(S^-1).

//   let (eigs, vecs) = rho.eigh(UPLO::Lower).unwrap();
//   let (eig_0,eig_1) = (eigs[0], eigs[1]);
//   let (vecs_0, vecs_1, vecs_2, vecs_3) = (vecs[[0 , 0]] , vecs[[0 , 1]] , vecs[[1 , 0]] , vecs[[1 , 1]]);

//   let mat_s = array![ [ vecs_0 , vecs_1 ], 
//                       [ vecs_2 , vecs_3 ] ];

//   let mat_d = array![ [ eig_0 ,   0.0   ], 
//                       [   0.0 , eig_1   ] ];

//   let mat_s_inv = mat_s.inv().unwrap();

//   let sqrt_mat_d = mat_d.ssqrt(UPLO::Lower).unwrap();
//   let sqrt_product = mat_s.dot(&sqrt_mat_d).dot(&mat_s_inv);
//   sqrt_product
// }
///////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_fibc() {
    let actual = fibc(Complex::new(2.1, -3.3));
    // let expected = Complex::new( 842.107900737978028312629,  -2448.9411071123121308522611};
    let expected = Complex::new(2.1, 4.4);
    
    assert!( (actual - expected).norm_sqr() <= 1e-6 );
    println!("{}" , actual)
  }
}

