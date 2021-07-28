use std::f64::consts::PI;
use ndarray::prelude::*;
use ndarray_linalg::*;
extern crate lazy_static;
extern crate ndarray;
use ndarray::{concatenate, Axis};
use lazy_static::lazy_static;

pub use ndarray_linalg::c64;
pub type VecC64 = ndarray::Array1<c64>;
pub type VecF64 = ndarray::Array1<f64>;
pub type MatrixC64 = ndarray::Array2<c64>;
pub type MatrixF64 = ndarray::Array2<f64>;


//TODO: Create a module of these comments
//TODO: Rust Doc comments in Rust cookbook
//TODO: Stack/concatenate
//TODO: Be able to read in csv of numpy arrays
//TODO: Optimize by using borrowed inputs for functions "&" instead of cloning
/////////////////////////////////////////////////////////////////////
////////////////////Entanglement Calculations////////////////////////
/////////////////////////////////////////////////////////////////////

pub fn create_dens_matrix(qm_state_vector: &VecC64) -> MatrixC64 {

  let qm_state_vector_conj = qm_state_vector.map(|qm_state_vector| qm_state_vector.conj());
  let a = into_col(qm_state_vector.clone());
  let b = into_row(qm_state_vector_conj);
  let dens_matrix = a.dot(&b);
  dens_matrix
}

pub fn find_purity(rho: MatrixC64)-> f64 {

  let rho_squared = rho.dot(&rho); 
  let purity = rho_squared.trace().unwrap();
  purity.re
}


pub fn find_fidelity(rho: MatrixC64, sigma: MatrixC64) -> f64 {

  let sqrt_rho = find_sqr_root_of_matrix(rho);
  let product = sqrt_rho.dot(&sigma).dot(&sqrt_rho);
  let sqrt_product = find_sqr_root_of_matrix(product);
  (sqrt_product.trace().unwrap()).re
}

pub fn find_concurrence(rho: MatrixC64) -> f64 {

  let pauli_y: MatrixC64 = array![ 
    [c64::new(0.0 , 0.0) , c64::new(0.0 , -1.0)] ,
    [c64::new(0.0 , 1.0) , c64::new(0.0 , 0.0)] 
  ];

  let pauli_y_tensor = find_tensor_product(pauli_y.clone() , pauli_y.clone());

  let rho_star = rho.mapv(|rho| rho.conj());
  let sqrt_rho = find_sqr_root_of_matrix(rho.clone());
  let rho_tilde = pauli_y_tensor.dot(&rho_star).dot(&pauli_y_tensor);

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

pub fn find_two_source_hom(signal: VecF64, idler: VecF64, jsa: MatrixC64, dt: f64) -> (f64, f64, f64) {

  let two_pi_c_dt = 2.*PI*3.0e+8_f64*dt;
  let signal_len = signal.len();
  let idler_len= idler.len();

  let mut rate_ss = 0.;
  let mut rate_ii = 0.;
  let mut rate_si = 0.;
  for j in 0..signal_len{
    let s_j_inv = 1./signal[j];

    for k in 0..idler_len{
      let a = jsa[[j, k]];
      let s_k_inv = 1./signal[k];

      for l in 0..signal_len{
        let c = jsa[[l, k]];

        let i_l_inv = 1./idler[l]; 
        let arg_ss = two_pi_c_dt*(s_j_inv - i_l_inv);
        let phase_ss = c64::new(0. , arg_ss).exp();

        for m in 0..idler_len{
          let i_m_inv = 1./idler[m];

          let arg_ii = two_pi_c_dt*(s_k_inv - i_m_inv);
          let phase_ii = c64::new(0. , arg_ii).exp();

          let arg_si = two_pi_c_dt*(s_j_inv - i_m_inv);
          let phase_si = c64::new(0. , arg_si).exp();

          let b = jsa[ [l, m] ];
          let d = jsa[ [j, m] ];
          let arg_1 = a*b;
          let arg_2 = c*d;

          let intf_ss = (arg_1 - phase_ss*arg_2)*0.5;
          let intf_ii = (arg_1 - phase_ii*arg_2)*0.5;
          let intf_si = (arg_1 - phase_si*arg_2)*0.5;
          rate_ss += (intf_ss.abs())*(intf_ss.abs());
          rate_ii += (intf_ii.abs())*(intf_ii.abs()); 
          rate_si += (intf_si.abs())*(intf_si.abs());                   
        }
      }
    }
  }
(rate_ss, rate_ii, rate_si)
}

/////////////////////////////////////////////////////////////////////
/////////////////////////Matrix Operations///////////////////////////
/////////////////////////////////////////////////////////////////////

pub fn find_dim(matrix: MatrixC64)-> i32 {
  let shape = matrix.dim();
  shape.1 as i32
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
      // println!("WARNING: While finding the sqrt of a matrix, the eigenvalue {:.32} was negative, but rounded to 0 and rescaled. \n", eigvals[j]);
      eigvals[j] = 0.0;
  
      j += 1;
    }
  }

  let eigvals_c64 = eigvals.map(|f| c64::new(*f, 0.0));

  let matrix_d = MatrixC64::from_diag(&eigvals_c64);
  let matrix_s = vecs;

  (matrix_d, matrix_s)
}



pub fn find_partial_transpose(matrix: MatrixC64) -> MatrixC64 {

  let dim = find_dim(matrix.clone()) as usize;

  let upper_left_block  = matrix.slice(s! [0..(dim / 2)   , 0..(dim / 2)  ] );
  let upper_right_block = matrix.slice(s! [0..(dim / 2)   , (dim / 2)..dim] );
  let lower_left_block  = matrix.slice(s! [(dim / 2)..dim , 0..(dim / 2)  ] );
  let lower_right_block = matrix.slice(s! [(dim / 2)..dim , (dim / 2)..dim] );

  let upper_right_block_transpose = upper_right_block.t();
  let lower_left_block_transpose = lower_left_block.t();

  let top = concatenate![Axis(1) , upper_left_block, upper_right_block_transpose];
  let bottom = concatenate![Axis(1) , lower_left_block_transpose, lower_right_block];
  let partial_transpose_matrix = concatenate![Axis(0) , top, bottom];

  partial_transpose_matrix

}

pub fn find_tensor_product(matrix_a: MatrixC64, matrix_b: MatrixC64) -> MatrixC64 {

  let dim = find_dim(matrix_a.clone()) as usize;
  let len = dim as i32;

  let first_block = matrix_b.mapv(|matrix_b| matrix_a[[0,0]]*matrix_b);
  let mut row_block = first_block; 
  
  for j in 1..len{
    let new_block = matrix_b.mapv(|matrix_b| (matrix_b)*matrix_a[[0, j as usize]]);
    row_block = concatenate![Axis(1) , row_block, new_block];
  }

  let top = row_block;
  let mut tensor_product = top;
  
  for i in 1..len{

    let first_block = matrix_b.mapv(|matrix_b| matrix_a[[i as usize,0]]*matrix_b);
    let mut row_block = first_block; 
  
    for j in 1..len{
      let new_block = matrix_b.mapv(|matrix_b| (matrix_b)*matrix_a[[i as usize, j as usize]]);
      row_block = concatenate![Axis(1) , row_block, new_block];
      }

    let new = row_block;
    tensor_product = concatenate![Axis(0) , tensor_product, new];
  }

  tensor_product

}

