use ndarray_linalg::c64;
use qm::*;
use ndarray::prelude::*;
// use ndarray_linalg::*;

fn main() {

////////////////////////////////////////////////////////////////////////
  // Global variables (Bell States, maximally mixed state)

  let norm_const = 1./2_f64.sqrt();
  let bell_phi_plus_coef = array![c64::new(norm_const , 0.0) , c64::new(0.0        , 0.0) , 
                                  c64::new(0.0        , 0.0) , c64::new(norm_const , 0.0) ];

  let bell_phi_minus_coef = array![c64::new(norm_const , 0.0) , c64::new(0.0         , 0.0) , 
                                   c64::new(0.0        , 0.0) , c64::new(-norm_const , 0.0) ];

  let bell_psi_plus_coef = array![c64::new(0.0        , 0.0) , c64::new(norm_const , 0.0) , 
                                  c64::new(norm_const , 0.0) , c64::new(0.0        , 0.0) ];

  let bell_psi_minus_coef = array![c64::new(0.0         , 0.0) , c64::new(norm_const , 0.0) , 
                                   c64::new(-norm_const , 0.0) , c64::new(0.0 ,        0.0) ];

  let rho_max_mixed_1_qbit = array![ [c64::new(0.5 , 0.0) , c64::new(0.0 , 0.0)] ,
                                     [c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0)] ];

  let rho_max_mixed_2_qbit = array![ [c64::new(0.25 , 0.0) ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                     [c64::new(0.0 , 0.0)  ,  c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                     [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0) ] ,
                                     [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0)] ];
////////////////////////////////////////////////////////////////////////
  //Fibonacci function tests

  // println!("The 20th Fibonacci number is {}", fib(20));
  // println!("Fib(2.1 -3.3i) is {}", fibc(Complex::new(2.1, im:-3.3}));

//////////////////////////////////////////////////////////////////////////  
  //Density Matrix tests

  let rho_bell_phi_plus = create_dens_matrix(bell_phi_plus_coef);
  let rho_bell_phi_minus = create_dens_matrix(bell_phi_minus_coef);
  let rho_bell_psi_plus = create_dens_matrix(bell_psi_plus_coef);
  let rho_bell_psi_minus = create_dens_matrix(bell_psi_minus_coef);

  let rho_bell_psi_minus_2 = rho_bell_psi_minus.clone();
  let rho_bell_psi_minus_3 = rho_bell_psi_minus_2.clone();

  // println!("dens matrix for phi_+ = \n {} \n", rho_bell_phi_plus);
  // println!("dens matrix for phi_- = \n {} \n", rho_bell_phi_minus);
  // println!("dens matrix for psi_+ = \n {} \n", rho_bell_psi_plus);
  // println!("dens matrix for psi_- = \n {} \n", rho_bell_psi_minus)

//////////////////////////////////////////////////////////////////////////  
  //Purity test
  
  println!("Dens matrix for a maximally mixed state {}: \n",rho_max_mixed_1_qbit);
  let rho_a_max_mixed_1_qbit = rho_max_mixed_1_qbit.clone();
  let rho_sqrd_1_qbit = find_dens_matrix_sqrd(rho_a_max_mixed_1_qbit);
  let rho_a_sqrd_1_qbit = rho_sqrd_1_qbit.clone();
  let rho_b_sqrd_1_qbit = rho_a_sqrd_1_qbit.clone();
  println!("Dens matrix squared is \n {} \n",rho_sqrd_1_qbit);
  println!("The purity is {} \n", find_purity(rho_a_sqrd_1_qbit));
  println!("The purity ranges from {} to {} \n", 1./find_dim(rho_b_sqrd_1_qbit), 1);

  // println!("Dens matrix for a pure state {}: \n",rho_bell_phi_plus);
  // let rho_bell_2 = rho_bell_phi_plus.clone();
  // let rho_bell_sqrd = find_dens_matrix_sqrd(rho_bell_2);
  // let rho_bell_sqrd_2 = rho_bell_sqrd.clone();
  // let rho_bell_sqrd_3 = rho_bell_sqrd_2.clone();
  // println!("Dens matrix squared is \n {} \n",rho_bell_sqrd);
  // println!("The purity is {} \n", find_purity(rho_bell_sqrd_2));
  // println!("The purity ranges from {} to {}", 1./find_dim(rho_bell_sqrd_3), 1)  

//////////////////////////////////////////////////////////////////////////  
  //Fidelity test

  let rho_max_a_2_qbit = rho_max_mixed_2_qbit.clone();
  let rho_max_b_2_qbit = rho_max_a_2_qbit.clone();
  let rho_max_c_2_qbit = rho_max_b_2_qbit.clone();
  let rho_max_d_2_qbit = rho_max_c_2_qbit.clone();
  let rho_max_e_2_qbit = rho_max_d_2_qbit.clone();


  println!("The fidelity for a maximally mixed state with itself is {} \n",fidelity(rho_max_mixed_2_qbit,rho_max_a_2_qbit));
  println!("The fidelity for a maximally mixed state with phi_+ is {} \n",fidelity(rho_max_b_2_qbit,rho_bell_phi_plus));
  println!("The fidelity for a maximally mixed state with phi_- is {} \n",fidelity(rho_max_c_2_qbit,rho_bell_phi_minus));  
  println!("The fidelity for a maximally mixed state with psi_+  is {} \n",fidelity(rho_max_d_2_qbit,rho_bell_psi_plus));  
  println!("The fidelity for a maximally mixed state with psi_-  is {} \n",fidelity(rho_max_e_2_qbit,rho_bell_psi_minus));  


  println!("The fidelity for psi_- with psi_-  is {} \n",fidelity(rho_bell_psi_minus_2,rho_bell_psi_minus_3)); 
}
