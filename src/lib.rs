use ndarray::prelude::*;
use ndarray_linalg::*;
extern crate lazy_static;
extern crate ndarray;
use ndarray::{stack, Axis};
// use lazy_static::lazy_static;

pub use ndarray_linalg::c64;
pub type VecC64 = ndarray::Array1<c64>;
pub type VecF64 = ndarray::Array1<f64>;
pub type MatrixC64 = ndarray::Array2<c64>;
pub type MatrixF64 = ndarray::Array2<f64>;

//TODO: Create a module of these comments
//TODO: Rust Doc comments in Rust cookbook
//TODO: Stack/concatenate
//TODO: Be able to read in csv of numpy arrays
/////////////////////////////////////////////////////////////////////
////////////////////Entanglement Calculations////////////////////////
/////////////////////////////////////////////////////////////////////

pub fn create_dens_matrix(coefs: &VecC64) -> MatrixC64 {

  let coefs_conj = coefs.map(|coefs| coefs.conj());
  let a = into_col(coefs.clone());
  let b = into_row(coefs_conj);
  let dens_matrix = a.dot(&b);
  dens_matrix
}

pub fn find_purity(rho_sqrd: MatrixC64)-> f64 {
  let purity = rho_sqrd.trace().unwrap();
  purity.re
}


pub fn find_fidelity(rho: MatrixC64, sigma: MatrixC64) -> f64 {

  let sqrt_rho = find_sqr_root_of_matrix(rho);
  let product = sqrt_rho.dot(&sigma).dot(&sqrt_rho);
  let sqrt_product = find_sqr_root_of_matrix(product);
  (sqrt_product.trace().unwrap()).re
}

pub fn find_concurrence(rho: MatrixC64) -> f64 {
  
  let pauli_y = array![ [  c64::new(0. , 0.)  ,  c64::new(0. , 0.) , c64::new(0. , 0.) ,  c64::new(-1. , 0.)  ] , 
                        [  c64::new(0. , 0.)  ,  c64::new(0. , 0.) , c64::new(1. , 0.) ,  c64::new(0. , 0.)   ] ,
                        [  c64::new(0. , 0.)  ,  c64::new(1. , 0.) , c64::new(0. , 0.) ,  c64::new(0. , 0.)   ] ,
                        [  c64::new(-1. , 0.) ,  c64::new(0. , 0.) , c64::new(0. , 0.) ,  c64::new(0. , 0.)   ] ];

  let rho_star = rho.mapv(|rho| rho.conj());
  let sqrt_rho = find_sqr_root_of_matrix(rho.clone());
  let rho_tilde = pauli_y.dot(&rho_star).dot(&pauli_y);

  let product = sqrt_rho.dot(&rho_tilde).dot(&sqrt_rho);
  let sqrt_product = find_sqr_root_of_matrix(product);

  let (eigvals, _eigvecs) = sqrt_product.eigh(UPLO::Lower).unwrap();
  let mut eigvals = eigvals.to_vec();
  eigvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
  0_f64.max(eigvals[3] - eigvals[2] - eigvals[1] - eigvals[0])
}

pub fn find_negativity(rho: MatrixC64) -> f64 {

  let trace_norm = find_trace_norm(rho);
  (trace_norm - 1.)/2.
}

pub fn find_trace_norm(rho: MatrixC64) -> f64 {

  let rho_partial_transpose = find_partial_transpose(rho);
  let rho_partial_transpose_star   = rho_partial_transpose.mapv(|rho_partial_transpose| rho_partial_transpose.conj());
  let rho_partial_transpose_dagger = rho_partial_transpose_star.t();
  
  let inner_product = (rho_partial_transpose_dagger).dot(&rho_partial_transpose);
  let partial_transpose_norm = find_sqr_root_of_matrix(inner_product); 
  let trace_norm =  partial_transpose_norm.trace().unwrap();
  trace_norm.re
}

pub fn find_log_negativity(rho: MatrixC64) -> f64 {
  let neg = find_negativity(rho);
  (2.*neg + 1.).log2()
}

/////////////////////////////////////////////////////////////////////
/////////////////////////Matrix Operations///////////////////////////
/////////////////////////////////////////////////////////////////////

pub fn find_dim(matrix: MatrixC64)-> i32 {
  let shape = matrix.dim();
  shape.1 as i32
}

pub fn find_matrix_sqrd(matrix: MatrixC64) -> MatrixC64 {
  let matrix_sqrd = matrix.dot(&matrix); 
  matrix_sqrd
}

pub fn find_sqr_root_of_matrix(matrix: MatrixC64) -> MatrixC64 {
  
  let (matrix_d, matrix_s) = rescale_neg_eigvals(matrix);
  let matrix_s_inv = matrix_s.inv().unwrap();
  let sqrt_matrix_d = matrix_d.mapv(|matrix_d| (matrix_d).sqrt());
  let sqrt_product = matrix_s.dot(&sqrt_matrix_d).dot(&matrix_s_inv);
  sqrt_product
}

pub fn rescale_neg_eigvals(rho: MatrixC64) -> (MatrixC64, MatrixC64) {
  
  let (mut eigvals, vecs) = rho.eigh(UPLO::Lower).unwrap();
  let eig_len = eigvals.len() as i32;

  let mut j = 0;
  for _ctr in 0..eig_len {
    if eigvals[j] < 0.0 {
      println!("WARNING: While finding the sqrt of a matrix, the eigenvalue {:.32} was negative, but rounded to 0 and rescaled. \n", eigvals[j]);
      eigvals[j] = 0.0;
  
      j += 1;
    }
  }

  let eigvals_c64 = eigvals.map(|f| c64::new(*f, 0.0));

  let matrix_d = MatrixC64::from_diag(&eigvals_c64);
  let matrix_s = vecs;

  (matrix_d, matrix_s)
}

pub fn find_schmidt_number(jsi: MatrixF64) -> f64 {
  let jsa = jsi.mapv(|jsi| jsi.sqrt());
  let (_u, s, _v_transpose) = jsa.svd(true , true).unwrap();
  let sum_eigvals_sqrd = s.mapv(|s| s*s).sum();
  let norm_const = 1./sum_eigvals_sqrd;
  let renormed_s = s.mapv(|s| s*(norm_const.sqrt()));
  let sum_eig_sqrd = renormed_s.mapv(|renormed_s| renormed_s.powf(4.)).sum();
  let k = 1./sum_eig_sqrd;
  k
}

pub fn find_partial_transpose(matrix: MatrixC64) -> MatrixC64 {

  let dim = find_dim(matrix.clone()) as usize;
  let mut partial_transpose_matrix = MatrixC64::zeros((dim , dim).f());

  let upper_left_block  = matrix.slice(s! [0..(dim / 2)   , 0..(dim / 2)  ] );
  let upper_right_block = matrix.slice(s! [0..(dim / 2)   , (dim / 2)..dim] );
  let lower_left_block  = matrix.slice(s! [(dim / 2)..dim , 0..(dim / 2)  ] );
  let lower_right_block = matrix.slice(s! [(dim / 2)..dim , (dim / 2)..dim] );

  let upper_right_block_transpose = upper_right_block.t();
  let lower_left_block_transpose = lower_left_block.t();

  let a = upper_left_block.slice(s! [0..(dim / 2), 0]);
  let b = upper_right_block_transpose.slice(s! [0..(dim / 2), 0]);
  let c = stack![Axis(0) , a, b];


  println!("a is {}, b is {}, c is {}",a,b, c);
  let mut i = 0;
  for _index_1 in 0..dim/2 {
    let mut j = 0;
    for _index_2 in 0..dim/2 {
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
