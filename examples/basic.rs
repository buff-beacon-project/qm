use std::f64::consts::PI;
use qm::*;

use ndarray::prelude::*;

extern crate lazy_static;
use lazy_static::lazy_static;

extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use std::fs::File;

  pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
  pub const THETA: f64 = PI*(30./180.);

lazy_static!{
  pub static ref BELL_PHI_PLUS_VEC: VecC64 = array![c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) ];

  pub static ref BELL_PHI_MINUS_VEC: VecC64 = array![c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) , c64::new(0.0 , 0.0) , c64::new(-NORM_CONST , 0.0) ];

  pub static ref BELL_PSI_PLUS_VEC: VecC64 = array![c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(0.0 , 0.0) ];

  pub static ref BELL_PSI_MINUS_VEC: VecC64 = array![c64::new(0.0 , 0.0) , c64::new(NORM_CONST , 0.0) , c64::new(-NORM_CONST , 0.0) , c64::new(0.0 , 0.0) ];

  pub static ref RHO_MAX_MIXED_1_QBIT: MatrixC64 = array![ 
    [c64::new(0.5 , 0.0) , c64::new(0.0 , 0.0)] ,
    [c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0)] 
  ];

  pub static ref RHO_MAX_MIXED_2_QBIT: MatrixC64 = array![ 
    [c64::new(0.25 , 0.0) ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
    [c64::new(0.0 , 0.0)  ,  c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
    [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0) ] ,
    [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0)] 
  ];



  pub static ref PSI_PART_ENTANGLED: VecC64 = array![c64::new(THETA.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(THETA.sin(), 0.0)];

}

pub fn main() -> Result<(), Box<dyn Error>> {

  // test_dens_matrix();
  // test_purity();
  // test_fidelity();
  // test_concurrence();
  // test_partial_transpose();
  // test_find_schmidt_number();

  let file = File::create("test.csv");
  let file = match file{
    Ok(f) => f,
    Err(e) => return Err(e.into())
  };
  let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
  writer.serialize_array2(&RHO_MAX_MIXED_2_QBIT)?;

  // dbg!(read_c64_array("JSA_data5.csv".into(), 5));

  // Ok(())
  Err("Error error".into())
}

pub fn test_dens_matrix() {

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_VEC);
  let rho_bell_phi_minus = create_dens_matrix(&BELL_PHI_MINUS_VEC);
  let rho_bell_psi_plus = create_dens_matrix(&BELL_PSI_PLUS_VEC);
  let rho_bell_psi_minus = create_dens_matrix(&BELL_PSI_MINUS_VEC);

  println!("dens matrix for phi_+ = \n {} \n", rho_bell_phi_plus);
  println!("dens matrix for phi_- = \n {} \n", rho_bell_phi_minus);
  println!("dens matrix for psi_+ = \n {} \n", rho_bell_psi_plus);
  println!("dens matrix for psi_- = \n {} \n", rho_bell_psi_minus);
}

pub fn test_purity() {

  let rho_max_mixed_1_qbit: MatrixC64 = array![ [c64::new(0.5 , 0.0) , c64::new(0.0 , 0.0)] ,
                                                [c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0)] ];

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_VEC);

  println!("The purity is {} \n", find_purity(rho_max_mixed_1_qbit.clone()));
  println!("The purity ranges from {} to {} \n", 1./(find_dim(rho_max_mixed_1_qbit.clone()) as f64), 1);
  println!("The purity is {} \n", find_purity(rho_bell_phi_plus.clone()));
  println!("The purity ranges from {} to {}", 1./(find_dim(rho_bell_phi_plus.clone()) as f64), 1);
}

pub fn test_fidelity() {

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_VEC);
  let rho_bell_phi_minus = create_dens_matrix(&BELL_PHI_MINUS_VEC);
  let rho_bell_psi_plus = create_dens_matrix(&BELL_PSI_PLUS_VEC);
  let rho_bell_psi_minus = create_dens_matrix(&BELL_PSI_MINUS_VEC);
  let rho_part_entangled = create_dens_matrix(&PSI_PART_ENTANGLED);

  println!("The fidelity for a maximally mixed state with itself is {} \n",find_fidelity(RHO_MAX_MIXED_2_QBIT.clone(),RHO_MAX_MIXED_2_QBIT.clone()));
  println!("The fidelity for a maximally mixed state with phi_+ is {} \n",find_fidelity(RHO_MAX_MIXED_2_QBIT.clone(),rho_bell_phi_plus.clone()));
  println!("The fidelity for a maximally mixed state with phi_- is {} \n",find_fidelity(RHO_MAX_MIXED_2_QBIT.clone(),rho_bell_phi_minus.clone()));  
  println!("The fidelity for a maximally mixed state with psi_+  is {} \n",find_fidelity(RHO_MAX_MIXED_2_QBIT.clone(),rho_bell_psi_plus.clone()));  
  println!("The fidelity for a maximally mixed state with psi_-  is {} \n",find_fidelity(RHO_MAX_MIXED_2_QBIT.clone(),rho_bell_psi_minus.clone()));  

  println!("The fidelity for psi_- with psi_-  is {} \n",find_fidelity(rho_bell_psi_minus.clone(),rho_bell_psi_minus.clone()));
  println!("The fidelity for psi_+ with psi_-  is {} \n",find_fidelity(rho_bell_psi_plus.clone(),rho_bell_psi_minus.clone()));   
  println!("Fidelity of cos({:.4})|00> + sin({:.4})|11> is {}", THETA, THETA, find_fidelity(rho_part_entangled.clone(), rho_part_entangled.clone()));   
}

pub fn test_concurrence() {

  let rho_max_mixed_2_qbit: MatrixC64 = array![ 
    [c64::new(0.25 , 0.0) ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ],
    [c64::new(0.0 , 0.0)  ,  c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ],
    [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0) ],
    [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0)] 
  ];

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_VEC);
  let rho_bell_phi_minus = create_dens_matrix(&BELL_PHI_MINUS_VEC);
  let rho_bell_psi_plus = create_dens_matrix(&BELL_PSI_PLUS_VEC);
  let rho_bell_psi_minus = create_dens_matrix(&BELL_PSI_MINUS_VEC);
  let rho_part_entangled = create_dens_matrix(&PSI_PART_ENTANGLED);

  println!("Concurrence of phi_+ is {}", find_concurrence(rho_bell_phi_plus));
  println!("Concurrence of phi_- is {}", find_concurrence(rho_bell_phi_minus));
  println!("Concurrence of psi_+ is {}", find_concurrence(rho_bell_psi_plus));
  println!("Concurrence of psi_- is {}", find_concurrence(rho_bell_psi_minus));

  println!("Concurrence of mixed state is {}", find_concurrence(rho_max_mixed_2_qbit));
  println!("Concurrence of cos({:.4})|00> + sin({:.4})|11> is {}", THETA, THETA,find_concurrence(rho_part_entangled));  
}

pub fn test_partial_transpose() {

  let test_matrix: MatrixC64 = array![ [c64::new(1.0 , 0.0)  , c64::new(2.0 , 0.0)  , c64::new(3.0 , 0.0)  , c64::new(4.0 , 0.0) ] ,
    [c64::new(5.0 , 0.0)  , c64::new(6.0 , 0.0)  , c64::new(7.0 , 0.0)  , c64::new(8.0 , 0.0) ] ,
    [c64::new(9.0 , 0.0)  , c64::new(10.0 , 0.0) , c64::new(11.0 , 0.0) , c64::new(12.0 , 0.0)] ,
    [c64::new(13.0 , 0.0) , c64::new(14.0 , 0.0) , c64::new(15.0 , 0.0) , c64::new(16.0 , 0.0)] 
  ];

  println!("{}", find_partial_transpose(test_matrix))
}

// pub fn test_find_schmidt_number () {

//   let jsi = read_c64_array("JSA_data5.csv".to_string(),5).unwrap();
//   println!("{}",find_schmidt_number(jsi));
// }

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
    println!("{}",array_read);
    Ok(array_read)
}

