use std::f64::consts::PI;
// use std::vec::*;
// use std::cmp;
use ndarray::prelude::*;
use ndarray_linalg::*;

pub type Complex = num_complex::Complex<f64>;
pub type VecC64 = ndarray::Array1<c64>;
pub type VecC32 = ndarray::Array1<c32>;
pub type MatrixC64 = ndarray::Array2<c64>;
pub type MatrixC32 = ndarray::Array2<c32>;
pub type MatrixF64 = ndarray::Array2<f64>;
pub type MatrixF32 = ndarray::Array2<f32>;

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

pub fn create_dens_matrix(coefs: VecC32) -> MatrixC32{

  let coef_num = coefs.len() as i32;
  let dens_mat_len = coef_num as usize;
  let mut dens_matrix = MatrixC32::zeros((dens_mat_len,dens_mat_len).f());

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


pub fn find_dens_matrix_sqrd(rho: MatrixC32) -> MatrixC32{
  let rho_sqrd = rho.dot(&rho); 
  rho_sqrd
}

pub fn find_purity(rho_sqrd: MatrixC32)-> f32{
  let purity = rho_sqrd.trace().unwrap();
  purity.re
}
pub fn find_dim(rho_sqrd: MatrixC32)-> f32{
  let shape = rho_sqrd.dim();
  shape.1 as f32
}

pub fn fidelity(rho: MatrixC32, sigma: MatrixC32) -> f32 {

  let sqrt_rho = rho.ssqrt(UPLO::Lower).unwrap();
  let product = sqrt_rho.dot(&sigma).dot(&sqrt_rho);
  let sqrt_product = product.ssqrt(UPLO::Lower).unwrap();
  (sqrt_product.trace().unwrap()).re
}

////////////////////////////////////////////////////////
// pub fn find_concurrence(rho: MatrixC32) -> f32{

//   let pauli_y = array![ [  c32::new(0. , 0.)  ,  c32::new(0. , 0.) , c32::new(0. , 0.) ,  c32::new(-1. , 0.)  ] , 
//                         [  c32::new(0. , 0.)  ,  c32::new(0. , 0.) , c32::new(1. , 0.) ,  c32::new(0. , 0.)   ] ,
//                         [  c32::new(0. , 0.)  ,  c32::new(1. , 0.) , c32::new(0. , 0.) ,  c32::new(0. , 0.)   ] ,
//                         [  c32::new(-1. , 0.) ,  c32::new(0. , 0.) , c32::new(0. , 0.) ,  c32::new(0. , 0.)   ] ];
//   println!("1) rho is \n {} \n ", rho);
  
//   // if rho[[0 , 3]]==rho[[3,0]]{
//   //     println!("They are equal")};


//   let sqrt_rho_lower = rho.ssqrt(UPLO::Lower);
//   println!("2) sqrt_root_rho_lower = \n {:?}\n ",sqrt_rho_lower);

//   // println!("rho1 = {}, rho2 = {}",rho[[0 , 3]], rho[[3,0]]);

//   let sqrt_rho = sqrt_rho_lower.unwrap();
//   // println!("3) sqrt_root_rho_lower = \n {:?}\n ",sqrt_rho_lower);
//   println!("3) sqrt_root_rho = \n {:?}\n ",sqrt_rho);  
//   let rho_star = rho.mapv(|rho| rho.conj());

//   let rho_tilde = pauli_y.dot(&rho_star).dot(&pauli_y);

//   let product = sqrt_rho.dot(&rho_tilde).dot(&sqrt_rho);
//   println!("4) product = \n {:?}\n ",product);  
//   let sqrt_product = product.ssqrt(UPLO::Lower).unwrap();
//   println!("5) sqrt_product = \n {:?}\n ",sqrt_product); 

//   let (eigvals, _eigvecs) = sqrt_product.eigh(UPLO::Lower).unwrap();
  
//   println!("eigs = {}",eigvals);
//   0_f32.max(eigvals[3] - eigvals[2] - eigvals[1] - eigvals[0])
// }
/////////////////////////////////////////////////////////////////
pub fn find_concurrence(rho: MatrixC32) -> f32{

  let pauli_y = array![ [  c32::new(0. , 0.)  ,  c32::new(0. , 0.) , c32::new(0. , 0.) ,  c32::new(-1. , 0.)  ] , 
                        [  c32::new(0. , 0.)  ,  c32::new(0. , 0.) , c32::new(1. , 0.) ,  c32::new(0. , 0.)   ] ,
                        [  c32::new(0. , 0.)  ,  c32::new(1. , 0.) , c32::new(0. , 0.) ,  c32::new(0. , 0.)   ] ,
                        [  c32::new(-1. , 0.) ,  c32::new(0. , 0.) , c32::new(0. , 0.) ,  c32::new(0. , 0.)   ] ];
  println!("1) rho is \n {} \n ", rho);

  let sqrt_rho = find_sqr_root_of_matrix(rho.clone());
  println!("3) sqrt_root_rho = \n {:?}\n ",sqrt_rho);  
  let rho_star = rho.mapv(|rho| rho.conj());

  let rho_tilde = pauli_y.dot(&rho_star).dot(&pauli_y);

  let product = sqrt_rho.dot(&rho_tilde).dot(&sqrt_rho);
  println!("4) product = \n {:?}\n ",product);  

  let sqrt_product = find_sqr_root_of_matrix(product);
  println!("5) sqrt_product = \n {:?}\n ",sqrt_product); 

  let (eigvals, _eigvecs) = sqrt_product.eigh(UPLO::Lower).unwrap();
  
  println!("eigs = {}",eigvals);
  0_f32.max(eigvals[3] - eigvals[2] - eigvals[1] - eigvals[0])
}

////////////////////////////////////////////////////////////////
pub fn find_sqr_root_of_matrix(rho: MatrixC32) -> MatrixC32 {

//Hermitian matrices are always diagonalizable, so we can express rho = S*D*(S^-1),
//so the sqrt(rho) = S*(D^0.5)*(S^-1). D^0.5 is a matrix of the sqr root of the 
//elements of D since it is diagonal.

  let (mut eigs, vecs) = rho.eigh(UPLO::Lower).unwrap();


  println!("vecs = {}",vecs);
  let mat_s = vecs.clone();

  let eig_len = eigs.len() as i32;

  let mut j = 0;
  for _ctr in 0..eig_len {
    if eigs[j] < 0.0 {
      eigs[j] = 0.0;
      println!("WARNING: The eivenvalue {} was negative, but rounded to 0", eigs[j]);
      j += 1;
    }
  }

  println!("eigs = {}",eigs);

  let (eig_0 , eig_1, eig_2, eig_3) = (c32::new(eigs[0], 0.0), c32::new(eigs[1], 0.0), c32::new(eigs[2], 0.0), c32::new(eigs[3], 0.0),);

  // println!("mat_s = {}", mat_s);
  let mat_d = array![ [ eig_0 ,              c32::new(0.0, 0.0) , c32::new(0.0, 0.0) , c32::new(0.0, 0.0) ] , 
                      [ c32::new(0.0, 0.0) , eig_1              , c32::new(0.0, 0.0) , c32::new(0.0, 0.0) ] , 
                      [ c32::new(0.0, 0.0) , c32::new(0.0, 0.0) , eig_2              , c32::new(0.0, 0.0) ] , 
                      [ c32::new(0.0, 0.0) , c32::new(0.0, 0.0) , c32::new(0.0, 0.0) , eig_3              ] ];

  // println!("mat_d = {}", mat_d);
  let det = mat_s.det().unwrap();
  // println!("det  = {}", det);
  let mat_s_inv = mat_s.inv().unwrap();
  // println!("mat_s_inv = {}", mat_s_inv);

  // println!("SDS_inv = {}", mat_s.dot(&mat_d).dot(&mat_s_inv));
  let sqrt_mat_d = mat_d.mapv(|mat_d| (mat_d.re).sqrt());
  let sqrt_mat_d_complex = sqrt_mat_d.map(|f| c32::new(*f, 0.0));
  let sqrt_product = mat_s.dot(&sqrt_mat_d_complex).dot(&mat_s_inv);
  sqrt_product
}
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

