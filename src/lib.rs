#![crate_name = "qm"]

use std::f64::consts::PI;
use ndarray::prelude::*;
use ndarray_linalg::{into_col, into_row, Trace, Eigh, UPLO, SVD, Inverse};
extern crate lazy_static;
extern crate ndarray;
use ndarray::{concatenate, Axis};

pub use ndarray_linalg::c64;
pub type VecC64 = ndarray::Array1<c64>;
pub type VecF64 = ndarray::Array1<f64>;
pub type MatrixC64 = ndarray::Array2<c64>;
pub type MatrixF64 = ndarray::Array2<f64>;

/// Creates the density matrix from a state column vector.
/// 
/// Takes the outer product of the qm state vector (ket) times the conjugate transpose of the state vector (bra).
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::create_density_matrix;
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
///
/// let qm_vec: VecC64 = array!
///   [c64::new(0.0, 0.0), c64::new(0.5, 0.5), c64::new(0.5, -0.5)];
///
/// let density_matrix = create_density_matrix(&qm_vec);
///
/// ```

pub fn create_density_matrix(qm_state_vector: &VecC64) -> MatrixC64 {

  let qm_state_vector_conj = qm_state_vector.map(|qm_state_vector| qm_state_vector.conj());
  let a = into_col(qm_state_vector.clone());
  let b = into_row(qm_state_vector_conj);
  let density_matrix = a.dot(&b);
  density_matrix
}

/// Computes the purity of a density matrix by taking the trace of the density matrix squared.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_dim, find_purity};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// 
/// pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
/// let bell_state_phi_plus = array![ c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) ];
/// let rho = create_density_matrix(&bell_state_phi_plus);
/// let purity = find_purity(rho.clone());
///
/// let dim = find_dim(&rho);
///
/// println!("The purity is {} \n", purity);
/// println!("The purity ranges from {} to {} \n", 1./(dim as f64), 1)
/// ```

pub fn find_purity(rho: MatrixC64)-> f64 {

  let rho_squared = rho.dot(&rho); 
  let purity = rho_squared.trace().unwrap();
  purity.re
}

/// Computes the fidelity between two density matrices. 
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_fidelity};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// 
/// pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
/// let bell_state_phi_minus = array![ c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(-NORM_CONST , 0.0) ];
/// let rho_1 = create_density_matrix(&bell_state_phi_minus);
///
/// let diag: VecC64 = Array::from_elem( 4, c64::new(1./4., 0.0) );
/// let rho_2: MatrixC64 = MatrixC64::from_diag(&diag);
/// 
/// let fidelity = find_fidelity(&rho_1, &rho_2);
/// ```

pub fn find_fidelity(rho_1: &MatrixC64, rho_2: &MatrixC64) -> f64 {

  let sqrt_rho_1 = find_symmetric_square_root(&rho_1);
  let product = sqrt_rho_1.dot(&rho_2.clone()).dot(&sqrt_rho_1);
  let sqrt_product = find_symmetric_square_root(&product);
  (sqrt_product.trace().unwrap()).re
}

/// Computes the concurrence of a density matrix. Note that this has been extended for 
/// any number of qubits, but the computation will return zero for any odd number of 
/// qubits.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_concurrence};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// use std::f64::consts::PI;
/// 
/// pub const THETA: f64 = PI*(30./180.);
/// pub const PHI: f64 = PI*(45./180.);
/// let bloch_state = array![c64::new(THETA.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(PHI.cos()*THETA.sin(), PHI.sin())];
/// let rho = create_density_matrix(&bloch_state);
/// 
/// let concurrence = find_concurrence(&rho);
/// ```

pub fn find_concurrence(rho: &MatrixC64) -> f64 {

  let pauli_y: MatrixC64 = array![ 
    [c64::new(0.0 , 0.0) , c64::new(0.0 , -1.0)] ,
    [c64::new(0.0 , 1.0) , c64::new(0.0 , 0.0) ] 
  ];

  let dim = find_dim(&rho) as f64;
  let n_qubits = dim.log2() as i32;

  let pauli_y_product = (1..n_qubits).fold(pauli_y.clone(), |prev, _i|{
                          find_tensor_product(&prev, &pauli_y)
                        });

  let rho_star = rho.mapv(|rho| rho.conj());
  let sqrt_rho = find_symmetric_square_root(&rho);
  let rho_tilde = pauli_y_product.dot(&rho_star).dot(&pauli_y_product);

  let product = sqrt_rho.dot(&rho_tilde).dot(&sqrt_rho);
  let sqrt_product = find_symmetric_square_root(&product);

  let (eigvals, _eigvecs) = sqrt_product.eigh(UPLO::Lower).unwrap();
  let mut eigvals = eigvals.to_vec();
  eigvals.sort_by(|a, b| a.partial_cmp(b).unwrap());

  let eigval_sum = (1..dim as i32 - 1).fold(eigvals[0], |prev, i|{
                          eigvals[i as usize] + prev
                        });

  0_f64.max(eigvals[dim as usize - 1] - eigval_sum)
}

/// Computes the trace norm of a density matrix.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_trace_norm};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// use std::f64::consts::PI;
/// 
/// pub const THETA: f64 = PI*(120./180.);
/// let qm_state = array![c64::new(THETA.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(THETA.sin(), 0.0)];
/// let rho = create_density_matrix(&qm_state);
/// 
/// let trace_norm = find_trace_norm(&rho);
/// ```

pub fn find_trace_norm(rho: &MatrixC64) -> f64 {

  let rho_star   = rho.mapv(|rho| rho.conj());
  let rho_dagger = rho_star.t();
  
  let product = (rho_dagger).dot(rho);
  let square_root_of_product = find_symmetric_square_root(&product); 
  let trace_norm =  square_root_of_product.trace().unwrap();
  trace_norm.re
}

/// Computes the negativity of a density matrix.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_negativity};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// 
/// pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
/// let bell_state_psi_plus = array![ c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) ];
/// let rho = create_density_matrix(&bell_state_psi_plus);
/// 
/// let negativity = find_negativity(&rho);
/// ```

pub fn find_negativity(rho: &MatrixC64) -> f64 {

  let rho_partial_transpose = find_partial_transpose(&rho);
  let trace_norm = find_trace_norm(&rho_partial_transpose);
  (trace_norm - 1.)/2.
}

/// Computes the log negativity of a density matrix.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_log_negativity};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// use std::f64::consts::PI;
/// 
/// pub const THETA: f64 = PI*(30./180.);
/// pub const PHI: f64 = PI*(45./180.);
/// let bloch_state = array![c64::new(THETA.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(PHI.cos()*THETA.sin(), PHI.sin())];
/// let rho_1 = create_density_matrix(&bloch_state);
/// 
/// let diag: VecC64 = Array::from_elem( 4, c64::new(1./4., 0.0) );
/// let rho_2: MatrixC64 = MatrixC64::from_diag(&diag);
/// 
/// let rho_mixed = c64::new(0.3, 0.0)*rho_1 + c64::new(0.7, 0.0)*rho_2;
/// let log_negativity = find_log_negativity(&rho_mixed);
/// ```

pub fn find_log_negativity(rho: &MatrixC64) -> f64 {
  let neg = find_negativity(&rho);
  (2.*neg + 1.).log2()
}

/// Computes the Schmidt number, or rank *k*, from a joint spectral intensity matrix (JSI).
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::find_schmidt_number;
/// use qm::{MatrixF64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// 
/// extern crate csv;
/// extern crate ndarray;
/// extern crate ndarray_csv;
/// use csv::{ReaderBuilder};
/// use ndarray_csv::{Array2Reader};
/// use std::error::Error;
/// use std::fs::File;
///
/// pub fn read_f64_array(csv_file: String, array_size: usize) -> Result<MatrixF64, Box<dyn Error>> {
///   let file = File::open(csv_file)?;
///   let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
///   let array_read: MatrixF64 = reader.deserialize_array2((array_size, array_size))?;
///   println!("{}",array_read);
///   Ok(array_read)
///   }
///
/// let jsi = read_f64_array("JSI_data200.csv".to_string(),200).unwrap();
/// println!("{}",find_schmidt_number(&jsi));
///
/// ```

pub fn find_schmidt_number(jsi: &MatrixF64) -> f64 {

  let jsa = jsi.mapv(|jsi| jsi.sqrt());
  let (_u, s, _v_transpose) = jsa.svd(true , true).unwrap();
  let sum_eigvals_sqrd = s.mapv(|s| s*s).sum();
  let norm_const = 1./sum_eigvals_sqrd;
  let renormed_s = s.mapv(|s| s*(norm_const.sqrt()));
  let sum_eig_sqrd = renormed_s.mapv(|renormed_s| renormed_s.powf(4.)).sum();
  let k = 1./sum_eig_sqrd;
  k
}

/// Used to graph the two-source Hong-Ou-Mandel effect.
/// 
/// Computes a coincidence probability for a certain time interval dt. The
/// plotters crate can be used in order to graph the data points.
/// 
/// # Examples
///
/// ```rust,editable
/// extern crate plotters;
/// extern crate ndarray;

/// pub use ndarray_linalg::c64;
/// use std::f64::consts::PI;
/// use plotters::prelude::*;
/// use ndarray::prelude::*;
/// use ndarray_linalg::*;
/// use qm::*;
/// 
/// extern crate csv;
/// extern crate ndarray_csv;
/// use csv::{ReaderBuilder};
/// use ndarray_csv::{Array2Reader};
/// use std::error::Error;
/// use std::fs::File;
///
/// pub fn read_c64_array(csv_file: String, array_size: usize) -> Result<MatrixC64, Box<dyn Error>> {
///     let file = File::open(csv_file)?;
///     let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
///     let array_read: MatrixC64 = reader.deserialize_array2((array_size, array_size))?;
///     Ok(array_read)
/// }
/// 
/// pub fn read_f64_vector(csv_file: String) -> Result<VecF64, Box<dyn Error>> {
/// 
///     let file = File::open(csv_file)?;
///     let mut reader = ReaderBuilder::new()
///         .from_reader(file);
/// 
///     let v : Vec<f64> = reader.headers()?
///         .into_iter()
///         .map(|s| s.parse::<f64>().unwrap()) // to f64
///         .collect(); // create a vec      
///     
///     let arr = Array::from(v);
///     Ok(arr)
/// }
///
/// let signal: VecF64 = read_f64_vector("signal_data5.csv".to_string()).unwrap();
/// let idler: VecF64 = read_f64_vector("idler_data5.csv".to_string()).unwrap();
/// let jsa: MatrixC64 = read_c64_array("jsa_data5.csv".to_string() , 5).unwrap();
/// let norm = two_source_hom_norm(&signal, &idler, &jsa);
/// let times = Array::linspace(-300.0e-15_f64, 300.0e-15_f64, 100);
///
/// let mut points_time: Vec<f64> = Vec::new();
/// let mut points_ss: Vec<f64> = Vec::new();
/// let mut points_ii: Vec<f64> = Vec::new();
/// 
/// for i in 0..99{
///     let rate = find_two_source_hom(&signal, &idler, &jsa, times[i]);
///     let rate_ss = rate.0/norm;
///     let rate_ii = rate.1/norm;
///     points_time.push(times[i]*1.0e+15_f64);
///     points_ss.push(rate_ss);
///     points_ii.push(rate_ii);
/// }
/// ```

pub fn find_two_source_hom(signal: &VecF64, idler: &VecF64, jsa: &MatrixC64, dt: f64) -> (f64, f64, f64) {

  let two_pi_c_dt = 2.*PI*3.0e+8_f64*dt;

  let mut rate_ss = 0.;
  let mut rate_ii = 0.;
  let mut rate_si = 0.;

  for (j, s_j) in signal.iter().enumerate() {
   let s_j_inv = 1. / s_j;

    for (k, s_k) in idler.iter().enumerate() {
      let a = jsa[[j, k]];
      let s_k_inv = 1. / s_k;

      for (l, i_l) in signal.iter().enumerate() {
        let c = jsa[[l, k]];
        let i_l_inv = 1. / i_l; 
        let arg_ss = two_pi_c_dt*(s_j_inv - i_l_inv);
        let phase_ss = ( c64::new(0. , 1.0) * c64::from(arg_ss) ).exp(); 
        
        for (m, i_m) in idler.iter().enumerate() {
          let i_m_inv = 1. / i_m;

          let arg_ii = two_pi_c_dt*(s_k_inv - i_m_inv);
          let phase_ii = ( c64::new(0. , 1.0) * c64::from(arg_ii) ).exp(); 

          let arg_si = two_pi_c_dt*(s_j_inv - i_m_inv);
          let phase_si = ( c64::new(0. , 1.0) * c64::from(arg_si) ).exp(); 


          let b = jsa[ [l, m] ];
          let d = jsa[ [j, m] ];
          let arg_1 = a*b;
          let arg_2 = c*d;

          let intf_ss = 0.5*(arg_1 - phase_ss*arg_2);
          let intf_ii = 0.5*(arg_1 - phase_ii*arg_2);
          let intf_si = 0.5*(arg_1 - phase_si*arg_2);
          rate_ss += intf_ss.norm_sqr();
          rate_ii += intf_ii.norm_sqr(); 
          rate_si += intf_si.norm_sqr();                   
        }
      }
    }
  }
(rate_ss, rate_ii, rate_si)
}

/// Used to normalize the coincidence probabilities while using the 
/// *find_two_source_hom* function.
/// 
/// # Examples
///
/// ```rust,editable
/// extern crate plotters;
/// extern crate ndarray;

/// pub use ndarray_linalg::c64;
/// use std::f64::consts::PI;
/// use plotters::prelude::*;
/// use ndarray::prelude::*;
/// use ndarray_linalg::*;
/// use qm::*;
/// 
/// extern crate csv;
/// extern crate ndarray_csv;
/// use csv::{ReaderBuilder};
/// use ndarray_csv::{Array2Reader};
/// use std::error::Error;
/// use std::fs::File;
///
/// pub fn read_c64_array(csv_file: String, array_size: usize) -> Result<MatrixC64, Box<dyn Error>> {
///     let file = File::open(csv_file)?;
///     let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
///     let array_read: MatrixC64 = reader.deserialize_array2((array_size, array_size))?;
///     Ok(array_read)
/// }
/// 
/// pub fn read_f64_vector(csv_file: String) -> Result<VecF64, Box<dyn Error>> {
/// 
///     let file = File::open(csv_file)?;
///     let mut reader = ReaderBuilder::new()
///         .from_reader(file);
/// 
///     let v : Vec<f64> = reader.headers()?
///         .into_iter()
///         .map(|s| s.parse::<f64>().unwrap()) // to f64
///         .collect(); // create a vec      
///     
///     let arr = Array::from(v);
///     Ok(arr)
/// }
///
/// let signal: VecF64 = read_f64_vector("signal_data5.csv".to_string()).unwrap();
/// let idler: VecF64 = read_f64_vector("idler_data5.csv".to_string()).unwrap();
/// let jsa: MatrixC64 = read_c64_array("jsa_data5.csv".to_string() , 5).unwrap();
/// let norm = two_source_hom_norm(&signal, &idler, &jsa);
/// ```

pub fn two_source_hom_norm(signal: &VecF64, idler: &VecF64, jsa: &MatrixC64)-> f64 {
    let mut rate = 0.;
    for (j, _s_j) in signal.iter().enumerate() {
        
        for (k, _s_k) in idler.iter().enumerate() {
            let a = jsa[ [j , k] ];
            
            for (l, _i_l) in signal.iter().enumerate() {
                
                for (m, _i_m) in idler.iter().enumerate() {
                    let b = jsa[ [l , m] ];
                    
                    let arg1 = a*b;
                    
                    rate += arg1.norm_sqr();
                }
            }
        }
    }
    rate
}

/// Computes the dimension (length) of a square matrix.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_dim};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// 
/// pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
/// let bell_state_phi_plus = array![ c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) ];
/// let rho = create_density_matrix(&bell_state_phi_plus);
///
/// let dim = find_dim(&rho);
/// ```

pub fn find_dim(matrix: &MatrixC64)-> i32 {
  let shape = matrix.dim();
  shape.1 as i32
}

/// Computes the symmetric square root of a Hermitian matrix. 

/// If there are numerical precision errors with computing negative eigenvalues while using LAPACK for
/// semi-positive definite, diagonalizable matrices, *rescale_neg_eigvals* will be called.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_symmetric_square_root};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// use std::f64::consts::PI;
/// 
/// pub const THETA: f64 = PI*(30./180.);
/// pub const PHI: f64 = PI*(45./180.);
/// let bloch_state = array![c64::new(THETA.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(PHI.cos()*THETA.sin(), PHI.sin())];
/// let rho_1 = create_density_matrix(&bloch_state);
/// 
/// let diag: VecC64 = Array::from_elem( 4, c64::new(1./4., 0.0) );
/// let rho_2: MatrixC64 = MatrixC64::from_diag(&diag);
/// 
/// let rho_mixed = c64::new(0.5, 0.0)*rho_1 + c64::new(0.5, 0.0)*rho_2;
/// let sqrt_rho_mixed = find_symmetric_square_root(&rho_mixed);
/// ```

pub fn find_symmetric_square_root(matrix: &MatrixC64) -> MatrixC64 {
  
  let (matrix_d, matrix_s) = rescale_neg_eigvals(&matrix);
  let matrix_s_inv = matrix_s.inv().unwrap();
  let sqrt_matrix_d = matrix_d.mapv(|matrix_d| (matrix_d).sqrt());
  let sqrt_product = matrix_s.dot(&sqrt_matrix_d).dot(&matrix_s_inv);
  sqrt_product
}

/// Rescales negative eigenvalues of a semi-positive definite, diagonal matrix to zero. The numerical 
/// precision of the negative eigenvalues are extremely small, on the order of 10^-16.
///
/// Sometimes, there are numerical precision errors with computing negative eigenvalues when calling
/// LAPACK, and the *rescale_neg_eigvals* function will rescale any negative eigenvalues to zero and 
/// will return a diagonal matrix D and invertible matrix S.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, rescale_neg_eigvals};
/// use qm::{VecC64 , MatrixC64};
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// 
/// pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
/// let bell_state_phi_plus = array![ c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) ];
/// let rho = create_density_matrix(&bell_state_phi_plus);
///
/// let (matrix_s, matrix_d) = rescale_neg_eigvals(&rho);

pub fn rescale_neg_eigvals(rho: &MatrixC64) -> (MatrixC64, MatrixC64) {
  
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

/// Computes the partial transpose of a matrix with even-numbered dimensions.
/// 
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_partial_transpose};
/// use qm::MatrixC64;
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
/// 
/// let matrix: MatrixC64 = array![ 
///   [c64::new(1.0 , 0.0)  , c64::new(2.0 , 0.0)  , c64::new(3.0 , 0.0)  , c64::new(4.0 , 0.0) ] ,
///   [c64::new(5.0 , 0.0)  , c64::new(6.0 , 0.0)  , c64::new(7.0 , 0.0)  , c64::new(8.0 , 0.0) ] ,
///   [c64::new(9.0 , 0.0)  , c64::new(10.0 , 0.0) , c64::new(11.0 , 0.0) , c64::new(12.0 , 0.0)] ,
///   [c64::new(13.0 , 0.0) , c64::new(14.0 , 0.0) , c64::new(15.0 , 0.0) , c64::new(16.0 , 0.0)] 
/// ];
///
/// let part_transposed_matrix = find_partial_transpose(&matrix);
/// ```

pub fn find_partial_transpose(matrix: &MatrixC64) -> MatrixC64 {

  let dim = find_dim(&matrix) as usize;

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

/// Computes the tensor product of a matrix.
/// 
/// In this example, an iterator is used to find the Hadamard tensor product for 
/// an 8 qubit system. 
///
/// # Examples
///
/// ```rust,editable
/// use qm::{create_density_matrix, find_tensor_product};
/// use qm::MatrixC64;
/// use ndarray::prelude::*;
/// use ndarray_linalg::c64;
///
/// pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
/// let hadamard_2_qbit: MatrixC64 = array![ 
///   [c64::new(NORM_CONST , 0.0)  , c64::new(NORM_CONST , 0.0) ] ,
///   [c64::new(NORM_CONST , 0.0)  , c64::new(-NORM_CONST , 0.0) ] , 
/// ];
///
/// let hadamard_8_qbit = (1..8).fold(hadamard_2_qbit.clone(), |prev, _i|{
///                              find_tensor_product(&prev, &hadamard_2_qbit)
///                        });
/// ```

pub fn find_tensor_product(matrix_a: &MatrixC64, matrix_b: &MatrixC64) -> MatrixC64 {

  let len = find_dim(&matrix_a) as usize;

  let first_block = matrix_b.mapv(|matrix_b| matrix_a[[0,0]]*matrix_b);
  let mut row_block = first_block; 
  
  for j in 1..len{
    let new_block = matrix_b.mapv(|matrix_b| (matrix_b)*matrix_a[[0, j]]);
    row_block = concatenate![Axis(1) , row_block, new_block];
  }

  let top = row_block;
  let mut tensor_product = top;
  
  for i in 1..len{

    let first_block = matrix_b.mapv(|matrix_b| matrix_a[[i,0]]*matrix_b);
    let mut row_block = first_block; 
  
    for j in 1..len{
      let new_block = matrix_b.mapv(|matrix_b| (matrix_b)*matrix_a[[i, j]]);
      row_block = concatenate![Axis(1) , row_block, new_block];
      }

    let new = row_block;
    tensor_product = concatenate![Axis(0) , tensor_product, new];
  }

  tensor_product

}

#[cfg(test)]
mod tests {
  use super::*;
  extern crate csv;
  extern crate ndarray_csv;
  use csv::{ReaderBuilder};
  use ndarray_csv::{Array2Reader};
  use std::error::Error;
  use std::fs::File;

  extern crate lazy_static;
  use lazy_static::lazy_static;

  extern crate approx;
  use approx::*;

  pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
  pub const THETA: f64 = PI*(30./180.);

  lazy_static!{

    pub static ref BELL_PHI_PLUS_VEC: VecC64 = array!
      [ c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) ];

    pub static ref BELL_PHI_MINUS_VEC: VecC64 = array!
      [ c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(-NORM_CONST , 0.0) ];

    pub static ref BELL_PSI_PLUS_VEC: VecC64 = array!
      [ c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) ];

    pub static ref BELL_PSI_MINUS_VEC: VecC64 = array!
      [ c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(-NORM_CONST , 0.0) , c64::new(0.0 , 0.0) ];

    pub static ref PSI_PART_ENTANGLED: VecC64 = array!
      [c64::new(THETA.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(THETA.sin(), 0.0)];

    pub static ref TEST_VEC: VecC64 = array!
      [c64::new(0.0, 0.0), c64::new(0.5, 0.5), c64::new(0.5, -0.5)];

  }

  pub fn read_f64_array(csv_file: String, array_size: usize) -> Result<MatrixF64, Box<dyn Error>> {

    let file = File::open(csv_file)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: MatrixF64 = reader.deserialize_array2((array_size, array_size))?;
    println!("{}",array_read);
    Ok(array_read)
  }

  pub fn read_c64_array(csv_file: String, array_size: usize) -> Result<MatrixC64, Box<dyn Error>> {

    let file = File::open(csv_file)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: MatrixC64 = reader.deserialize_array2((array_size, array_size))?;
    Ok(array_read)
  }

  // pub fn read_f64_vector(csv_file: String) -> Result<VecF64, Box<dyn Error>> {

  //   let file = File::open(csv_file)?;
  //   let mut reader = ReaderBuilder::new()
  //       .from_reader(file);

  //   let v : Vec<f64> = reader.headers()?
  //       .into_iter()
  //       .map(|s| s.parse::<f64>().unwrap()) // to f64
  //       .collect(); // create a vec      
    
  //   let arr = Array::from(v);
  //   Ok(arr)
  // }

  pub fn assert_matrix_approx_equal(matrix1: &MatrixC64, matrix2: &MatrixC64, error: f64) -> (){

    let all_tests_good = matrix1.iter()
                                .zip(matrix2)
                                .map(|(el1, el2)|{
                                  abs_diff_eq!(el1.re, el2.re, epsilon = error) && 
                                  abs_diff_eq!(el1.im, el2.im, epsilon = error)
                                })
                                .all(|check| check);
    assert!(all_tests_good)
  }


  #[test]
      fn test_create_density_matrix() {

        let rho_bell_phi_plus = create_density_matrix(&BELL_PHI_PLUS_VEC);

        let rho_bell_phi_plus_actual = array![ 
          [c64::new(0.5 , 0.0) ,  c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0) ] ,
          [c64::new(0.0 , 0.0) ,  c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) ] ,
          [c64::new(0.0 , 0.0) ,  c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) ] ,
          [c64::new(0.5 , 0.0) ,  c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0) ] 
        ];

        assert_matrix_approx_equal(&rho_bell_phi_plus, &rho_bell_phi_plus_actual, 1.0e-06_f64)
      }

  #[test]
      fn test_find_purity() {

        let rho_bell_phi_minus = create_density_matrix(&BELL_PHI_MINUS_VEC);
        let purity_1 = find_purity(rho_bell_phi_minus);
        let bool_1 = abs_diff_eq!(purity_1 , 1.0 , epsilon = 1.0e-6_f64);

        let rho_mixed_diag: VecC64 = Array::from_elem( 16, c64::new(1./16., 0.0) );
        let rho_mixed: MatrixC64 = MatrixC64::from_diag(&rho_mixed_diag);
        let purity_2 = find_purity(rho_mixed);
        let bool_2 = abs_diff_eq!(purity_2 , 1./16. , epsilon = 1.0e-6_f64);

        assert!(bool_1);
        assert!(bool_2);
      }

  #[test]
      fn test_find_fidelity() {

        let rho_bell_psi_plus = create_density_matrix(&BELL_PSI_PLUS_VEC);
        let rho_bell_psi_minus = create_density_matrix(&BELL_PSI_MINUS_VEC);      
        let rho_mixed_diag: VecC64 = Array::from_elem( 4, c64::new(1./4., 0.0) );
        let rho_mixed: MatrixC64 = MatrixC64::from_diag(&rho_mixed_diag);

        let bool_1 = abs_diff_eq!(find_fidelity(&rho_bell_psi_plus, &rho_bell_psi_plus), 1.0, epsilon = 1.0e-6_f64);
        let bool_2 = abs_diff_eq!(find_fidelity(&rho_bell_psi_plus, &rho_mixed), 0.5, epsilon = 1.0e-6_f64);
        let bool_3 = abs_diff_eq!(find_fidelity(&rho_mixed, &rho_mixed), 1.0, epsilon = 1.0e-6_f64);
        let bool_4 = abs_diff_eq!(find_fidelity(&rho_bell_psi_plus, &rho_bell_psi_minus), 0.0, epsilon = 1.0e-6_f64);

        assert!(bool_1);
        assert!(bool_2);
        assert!(bool_3);
        assert!(bool_4);
      }

  #[test]
      fn test_find_concurrence() {

        let rho_bell_phi_plus = create_density_matrix(&BELL_PSI_PLUS_VEC);    
        let bool_1 = abs_diff_eq!(find_concurrence(&rho_bell_phi_plus), 1.0, epsilon = 1.0e-6_f64);

        let rho_mixed_diag: VecC64 = Array::from_elem( 64, c64::new(1./64., 0.0) );
        let rho_mixed: MatrixC64 = MatrixC64::from_diag(&rho_mixed_diag);     
        let bool_2 = abs_diff_eq!(find_concurrence(&rho_mixed), 0.0, epsilon = 1.0e-6_f64);       

        let rho_part_entangled = create_density_matrix(&PSI_PART_ENTANGLED);
        let bool_3 = abs_diff_eq!(find_concurrence(&rho_part_entangled), 3_f64.sqrt()/2., epsilon = 1.0e-6_f64); 

        assert!(bool_1);
        assert!(bool_2);
        assert!(bool_3);     
      }

  #[test]
      fn test_find_trace_norm () {

        let rho_bell_phi_minus = create_density_matrix(&BELL_PSI_MINUS_VEC);   
        let bool_1 = abs_diff_eq!(find_trace_norm(&rho_bell_phi_minus), 1.0, epsilon = 1.0e-6_f64);      
        assert!(bool_1);

        let rho_mixed_diag: VecC64 = Array::from_elem( 64, c64::new(1./64., 0.0) );
        let rho_mixed: MatrixC64 = MatrixC64::from_diag(&rho_mixed_diag);     
        let bool_2 = abs_diff_eq!(find_trace_norm(&rho_mixed), 1.0, epsilon = 1.0e-6_f64);
        assert!(bool_2);      
      }

  #[test]    
      fn test_find_negativity () {

        let rho_bell_phi_minus = create_density_matrix(&BELL_PHI_MINUS_VEC);   
        let bool_1 = abs_diff_eq!(find_negativity(&rho_bell_phi_minus), 0.5, epsilon = 1.0e-6_f64);      
        assert!(bool_1);

        let rho_mixed_diag: VecC64 = Array::from_elem( 32, c64::new(1./32., 0.0) );
        let rho_mixed: MatrixC64 = MatrixC64::from_diag(&rho_mixed_diag);     
        let bool_2 = abs_diff_eq!(find_negativity(&rho_mixed), 0.0, epsilon = 1.0e-6_f64);
        assert!(bool_2); 
      }

  #[test]  
      fn test_find_log_negativity() {

        let rho_bell_psi_plus = create_density_matrix(&BELL_PSI_PLUS_VEC);   
        let bool_1 = abs_diff_eq!(find_log_negativity(&rho_bell_psi_plus), 1.0, epsilon = 1.0e-6_f64);      
        assert!(bool_1);

        let rho_mixed_diag: VecC64 = Array::from_elem( 8, c64::new(1./8., 0.0) );
        let rho_mixed: MatrixC64 = MatrixC64::from_diag(&rho_mixed_diag);     
        let bool_2 = abs_diff_eq!(find_log_negativity(&rho_mixed), 0.0, epsilon = 1.0e-6_f64);
        assert!(bool_2); 
      }

  #[test]  
      fn test_find_schmidt_number() {    

        let jsi = read_f64_array("JSI_data200.csv".to_string() , 200).unwrap();
        let bool_1 = abs_diff_eq!(find_schmidt_number(&jsi), 6.738711627285404, epsilon = 1.0e-6_f64); 
        assert!(bool_1);     
      }

  #[test] 
      fn test_two_source_hom_norm(){

        let signal: VecF64 = array![1.4446824684143E-06,1.49734123420715E-06,1.54999999999999E-06,1.60265876579284E-06,1.65531753158569E-06];
        let idler: VecF64 = array![1.45728222027807E-06,1.51093173528974E-06,1.56458125030141E-06,1.61823076531308E-06,1.67188028032476E-06];
        let jsa: MatrixC64 = read_c64_array("jsa_data5.csv".to_string(),5).unwrap();

        assert_eq!(two_source_hom_norm(&signal, &idler, &jsa), 6.413926880524795e+61_f64);   
      }

  #[test] 
      fn test_find_dim() {
        
        let rho = create_density_matrix(&TEST_VEC);
        assert_eq!(find_dim(&rho) , 3);
      }

  #[test] 
      fn test_symmetric_square_root() {

        let rho_expected = create_density_matrix(&TEST_VEC);
        let sqrt_rho_expected = find_symmetric_square_root(&rho_expected);

        let sqrt_rho_actual = array![ 
          [c64::new(0.0 , 0.0) ,  c64::new(0.0 ,  0.0) , c64::new(0.0 , 0.0) ],
          [c64::new(0.0 , 0.0) ,  c64::new(0.5 ,  0.0) , c64::new(0.0 , 0.5) ],
          [c64::new(0.0 , 0.0) ,  c64::new(0.0 , -0.5) , c64::new(0.5 , 0.0) ],
        ];

        assert_matrix_approx_equal(&sqrt_rho_expected, &sqrt_rho_actual, 1.0e-06_f64)
      }

  #[test] 
  fn test_find_partial_transpose () {

    let matrix: MatrixC64 = array![ 
      [c64::new(1.0 , 0.0)  , c64::new(2.0 , 0.0)  , c64::new(3.0 , 0.0)  , c64::new(4.0 , 0.0) ] ,
      [c64::new(5.0 , 0.0)  , c64::new(6.0 , 0.0)  , c64::new(7.0 , 0.0)  , c64::new(8.0 , 0.0) ] ,
      [c64::new(9.0 , 0.0)  , c64::new(10.0 , 0.0) , c64::new(11.0 , 0.0) , c64::new(12.0 , 0.0)] ,
      [c64::new(13.0 , 0.0) , c64::new(14.0 , 0.0) , c64::new(15.0 , 0.0) , c64::new(16.0 , 0.0)] 
    ];

    let partial_transpose_expected = find_partial_transpose(&matrix);

    let partial_transpose_actual = array![ 
      [c64::new(1.0 , 0.0)  , c64::new(2.0 , 0.0)  , c64::new(3.0 , 0.0)  , c64::new(7.0 , 0.0) ] ,
      [c64::new(5.0 , 0.0)  , c64::new(6.0 , 0.0)  , c64::new(4.0 , 0.0)  , c64::new(8.0 , 0.0) ] ,
      [c64::new(9.0 , 0.0)  , c64::new(13.0 , 0.0) , c64::new(11.0 , 0.0) , c64::new(12.0 , 0.0)] ,
      [c64::new(10.0 , 0.0) , c64::new(14.0 , 0.0) , c64::new(15.0 , 0.0) , c64::new(16.0 , 0.0)] 
    ];

    assert_matrix_approx_equal(&partial_transpose_expected, &partial_transpose_actual, 1.0e-06_f64)    
  }

  #[test] 
  fn test_tensor_product(){

    let n_qbit = 8;
    let diag: VecC64 = Array::from_elem( 256, c64::new(1./256., 0.0) );
    let rho_8qbit_mixed_actual: MatrixC64 = MatrixC64::from_diag(&diag); 

    let rho_2qbit_mixed = array![ 
      [c64::new(0.5 , 0.0) , c64::new(0.0 , 0.0)],
      [c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0)] 
    ];

    let rho_8qbit_mixed_expected = (1..n_qbit).fold(rho_2qbit_mixed.clone(), |prev, _i|{
                                      find_tensor_product(&prev, &rho_2qbit_mixed)
                                        });
    assert_matrix_approx_equal(&rho_8qbit_mixed_expected, &rho_8qbit_mixed_actual, 1.0e-06_f64)                                              
  }

}