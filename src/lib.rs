use std::f64::consts::PI;
use ndarray::prelude::*;
use ndarray_linalg::*;

pub type Complex = num_complex::Complex<f64>;
pub type VecC64 = ndarray::Array1<c64>;
pub type MatrixC64 = ndarray::Array2<c64>;

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

pub fn create_dens_matrix(coefs: VecC64) -> MatrixC64{

  let coef_num = coefs.len() as i32;
  let dens_matrix_len = coef_num as usize;
  let mut dens_matrix = MatrixC64::zeros((dens_matrix_len , dens_matrix_len).f());

  let mut ctr_i = 0;

  for _coef in 0..coef_num{
    let mut ctr_j = 0;
    for _coef_2 in 0..coef_num{
      dens_matrix[ [ctr_i , ctr_j] ] = coefs[ctr_i] * ( coefs[ctr_j] ).conj();
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
pub fn find_dim(rho_sqrd: MatrixC64)-> i32 {
  let shape = rho_sqrd.dim();
  shape.1 as i32
}

pub fn fidelity(rho: MatrixC64, sigma: MatrixC64) -> f64 {

  let (matrix_d, matrix_s) = rescale_neg_eigvals(rho);
  let sqrt_rho = find_sqr_root_of_matrix(matrix_d, matrix_s);
  let product = sqrt_rho.dot(&sigma).dot(&sqrt_rho);
  let (matrix_d_product, matrix_s_product) = rescale_neg_eigvals(product);  
  let sqrt_product = find_sqr_root_of_matrix(matrix_d_product, matrix_s_product);
  (sqrt_product.trace().unwrap()).re
}

// pub fn fidelity_test(rho: MatrixC64, sigma: MatrixC64) -> f64 {

//   let (eigs_old, vecs_old) = rho.eigh(UPLO::Lower).unwrap();
//   println!("Eigenvalues of old matrix are {}",eigs_old);
//   println!("Eigenvectors of old matrix are {}",vecs_old);  
//   println!("rho_old is {}",rho);
//   let rho_rescaled = rescale_neg_eigvals(rho);
//   println!("rho_rescaled is {}",rho_rescaled);
//   let (eigs, vecs) = rho_rescaled.eigh(UPLO::Lower).unwrap();
//   println!("Eigenvalues of rescaled matrix are {}",eigs);
//   println!("Eigenvectors of rescaled matrix are {}",vecs);
//   let sqrt_rho = rho_rescaled.ssqrt(UPLO::Lower).unwrap();
//   let product = sqrt_rho.dot(&sigma).dot(&sqrt_rho);
//   let product_rescaled = rescale_neg_eigvals(product);  
//   let sqrt_product = product_rescaled.ssqrt(UPLO::Lower).unwrap();
//   (sqrt_product.trace().unwrap()).re
// }

pub fn find_concurrence(rho: MatrixC64) -> f64{

  let pauli_y = array![ [  c64::new(0. , 0.)  ,  c64::new(0. , 0.) , c64::new(0. , 0.) ,  c64::new(-1. , 0.)  ] , 
                        [  c64::new(0. , 0.)  ,  c64::new(0. , 0.) , c64::new(1. , 0.) ,  c64::new(0. , 0.)   ] ,
                        [  c64::new(0. , 0.)  ,  c64::new(1. , 0.) , c64::new(0. , 0.) ,  c64::new(0. , 0.)   ] ,
                        [  c64::new(-1. , 0.) ,  c64::new(0. , 0.) , c64::new(0. , 0.) ,  c64::new(0. , 0.)   ] ];

  let rho_star = rho.mapv(|rho| rho.conj());
  let (matrix_d, matrix_s) = rescale_neg_eigvals(rho.clone());
  let sqrt_rho = find_sqr_root_of_matrix(matrix_d, matrix_s);

  let rho_tilde = pauli_y.dot(&rho_star).dot(&pauli_y);

  let matrix_prod = sqrt_rho.dot(&rho_tilde).dot(&sqrt_rho);
  let (matrix_d_prod, matrix_s_prod) = rescale_neg_eigvals(matrix_prod);
  let sqrt_prod = find_sqr_root_of_matrix(matrix_d_prod, matrix_s_prod);

  let (eigvals, _eigvecs) = sqrt_prod.eigh(UPLO::Lower).unwrap();
  
  0_f64.max(eigvals[3] - eigvals[2] - eigvals[1] - eigvals[0])
}

pub fn find_negativity(rho: MatrixC64) -> f64{

  let trace_norm = find_trace_norm(rho);
  (trace_norm - 1.)/2.
}

pub fn find_trace_norm(rho: MatrixC64) -> f64{

  let rho_partial_transpose = find_partial_transpose(rho);
  let rho_partial_transpose_star   = rho_partial_transpose.mapv(|rho_partial_transpose| rho_partial_transpose.conj());
  let rho_partial_transpose_dagger = rho_partial_transpose_star.t();
  
  let inner_product = (rho_partial_transpose_dagger).dot(&rho_partial_transpose);
  let (matrix_d, matrix_s) = rescale_neg_eigvals(inner_product);
  let partial_transpose_norm = find_sqr_root_of_matrix(matrix_d, matrix_s); 
  let trace_norm =  partial_transpose_norm.trace().unwrap();
  trace_norm.re
}

pub fn find_partial_transpose(rho: MatrixC64) -> MatrixC64{

  let dim = find_dim(rho.clone()) as usize;
  let mut partial_transpose_matrix = MatrixC64::zeros((dim , dim).f());

  let upper_left_block  = rho.slice(s! [0..(dim / 2)   , 0..(dim / 2)  ] );
  let upper_right_block = rho.slice(s! [0..(dim / 2)   , (dim / 2)..dim] );
  let lower_left_block  = rho.slice(s! [(dim / 2)..dim , 0..(dim / 2)  ] );
  let lower_right_block = rho.slice(s! [(dim / 2)..dim , (dim / 2)..dim] );

  let upper_right_block_transpose = upper_right_block.t();
  let lower_left_block_transpose = lower_left_block.t();

  let mut i = 0;
  for _index_1 in 0..dim/2{
    let mut j = 0;
    for _index_2 in 0..dim/2{
      partial_transpose_matrix[[i         , j        ]] = upper_left_block[ [i , j] ];
      partial_transpose_matrix[[i         , j + dim/2]] = upper_right_block_transpose[ [i , j] ];
      partial_transpose_matrix[[i + dim/2 , j        ]] = lower_left_block_transpose[ [i , j] ];
      partial_transpose_matrix[[i + dim/2 , j + dim/2]] = lower_right_block[ [i , j] ];
      j += 1;
    }
    i += 1;
  }

  partial_transpose_matrix
}

pub fn find_log_negativity(rho: MatrixC64) -> f64{
  let neg = find_negativity(rho);
  (2.*neg + 1.).log2()
}


pub fn find_sqr_root_of_matrix(matrix_d: MatrixC64, matrix_s: MatrixC64) -> MatrixC64 {

//Hermitian matrices are always diagonalizable, so we can express rho = S*D*(S^-1),
//so the sqrt(rho) = S*(D^0.5)*(S^-1). D^0.5 is a matrix of the sqr root of the 
//elements of D since it is diagonal.

  let matrix_s_inv = matrix_s.inv().unwrap();
  let sqrt_matrix_d = matrix_d.mapv(|matrix_d| (matrix_d.re).sqrt());
  let sqrt_matrix_d_complex = sqrt_matrix_d.map(|f| c64::new(*f, 0.0));
  let sqrt_product = matrix_s.dot(&sqrt_matrix_d_complex).dot(&matrix_s_inv);
  sqrt_product

}

pub fn rescale_neg_eigvals(rho: MatrixC64) -> (MatrixC64, MatrixC64) {
  
  let (mut eigvals, vecs) = rho.eigh(UPLO::Lower).unwrap();
  let eig_len = eigvals.len() as i32;
  println!("eigvals = {}",eigvals);


  let mut j = 0;
  for _ctr in 0..eig_len {
    if eigvals[j] < 0.0 {
      println!("WARNING: While finding the sqrt of a matrix, the eigenvalue {:.32} was negative, but rounded to 0 and rescaled. \n", eigvals[j]);
      eigvals[j] = 0.0;
  
      j += 1;
    }
  }

  println!("eigvals_new = {}",eigvals);
  let eig_num = eigvals.len() as i32;
  let matrix_d_len = eig_num as usize;
  let mut matrix_d = MatrixC64::zeros((matrix_d_len,matrix_d_len).f());

  let mut i = 0;
  let eigvals_c64 = eigvals.map(|f| c64::new(*f, 0.0));

  for _num in 0..eig_num{
    let mut j = 0;
    for _num_2 in 0..eig_num{
      if i==j{
        matrix_d[ [i , j] ] = eigvals_c64[i];
      }
      j += 1;
    }
    i +=1;
  }

  let matrix_s = vecs;

  (matrix_d, matrix_s)

  //Used for debugging
  // let matrix_s_inv = matrix_s.inv().unwrap();
  // let rho_new = matrix_s.dot(&matrix_d).dot(&matrix_s_inv);
  // println!("matrix_d = {}", matrix_d);
  // let tr = rho_new.trace().unwrap();
  // let rho_rescaled = rho_new.mapv(|rho_new| rho_new/tr);
  // rho_new
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

