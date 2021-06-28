use ndarray_linalg::*;
use std::f64::consts::PI;
use qm::*;
use ndarray::prelude::*;

fn main() {

////////////////////////////////////////////////////////////////////////
  //Global variables (Bell States, maximally mixed state)

  // let norm_const = 1./2_f64.sqrt();
  // let bell_phi_plus_coef = array![c64::new(norm_const , 0.0) , c64::new(0.0        , 0.0) , 
  //                                 c64::new(0.0        , 0.0) , c64::new(norm_const , 0.0) ];

  // let bell_phi_minus_coef = array![c64::new(norm_const , 0.0) , c64::new(0.0         , 0.0) , 
  //                                  c64::new(0.0        , 0.0) , c64::new(-norm_const , 0.0) ];

  // let bell_psi_plus_coef = array![c64::new(0.0        , 0.0) , c64::new(norm_const , 0.0) , 
  //                                 c64::new(norm_const , 0.0) , c64::new(0.0        , 0.0) ];

  // let bell_psi_minus_coef = array![c64::new(0.0         , 0.0) , c64::new(norm_const , 0.0) , 
  //                                  c64::new(-norm_const , 0.0) , c64::new(0.0 ,        0.0) ];

  // // let rho_max_mixed_1_qbit = array![ [c64::new(0.5 , 0.0) , c64::new(0.0 , 0.0)] ,
  // //                                    [c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0)] ];

  let rho_max_mixed_2_qbit = array![ [c64::new(0.25 , 0.0) ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                     [c64::new(0.0 , 0.0)  ,  c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                     [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0) ] ,
                                     [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0)] ];

  let test_mat = array![             [c64::new(1.0 , 0.0) , c64::new(2.0 , 0.0)   , c64::new(3.0 , 0.0)  , c64::new(4.0 , 0.0) ] ,
                                     [c64::new(5.0 , 0.0) , c64::new(6.0 , 0.0)   , c64::new(7.0 , 0.0)  , c64::new(8.0 , 0.0) ] ,
                                     [c64::new(9.0 , 0.0) , c64::new(10.0 , 0.0)  , c64::new(11.0 , 0.0) , c64::new(12.0 , 0.0) ] ,
                                     [c64::new(13.0 , 0.0) , c64::new(14.0 , 0.0) , c64::new(15.0 , 0.0) , c64::new(16.0 , 0.0)] ];
  // let theta = PI*(60./180.);
  // let psi_part_entangled = array![c64::new(theta.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(theta.sin(), 0.0)];

////////////////////////////////////////////////////////////////////////
  //Fibonacci function tests

  // println!("The 20th Fibonacci number is {}", fib(20));
  // println!("Fib(2.1 -3.3i) is {}", fibc(Complex::new(2.1, im:-3.3}));

//////////////////////////////////////////////////////////////////////////  
  //Density Matrix tests

  // let rho_bell_phi_plus = create_dens_matrix(bell_phi_plus_coef);
  // let rho_bell_phi_minus = create_dens_matrix(bell_phi_minus_coef);
  // let rho_bell_psi_plus = create_dens_matrix(bell_psi_plus_coef);
  // let rho_bell_psi_minus = create_dens_matrix(bell_psi_minus_coef);

  // let rho_part_entangled = create_dens_matrix(psi_part_entangled);

  // println!("dens matrix for phi_+ = \n {} \n", rho_bell_phi_plus);
  // println!("dens matrix for phi_- = \n {} \n", rho_bell_phi_minus);
  // println!("dens matrix for psi_+ = \n {} \n", rho_bell_psi_plus);
  // println!("dens matrix for psi_- = \n {} \n", rho_bell_psi_minus);

  // println!("pauli spin = {}",pauli_tensor);
//////////////////////////////////////////////////////////////////////////  
  //Purity test
  
  // println!("Dens matrix for a maximally mixed state {}: \n",rho_max_mixed_1_qbit);
  // let rho_a_max_mixed_1_qbit = rho_max_mixed_1_qbit.clone();
  // let rho_sqrd_1_qbit = find_dens_matrix_sqrd(rho_a_max_mixed_1_qbit);
  // println!("Dens matrix squared is \n {} \n",rho_sqrd_1_qbit);
  // println!("The purity is {} \n", find_purity(rho_sqrd_1_qbit.clone()));
  // println!("The purity ranges from {} to {} \n", 1./find_dim(rho_sqrd_1_qbit.clone()), 1);

  // println!("Dens matrix for a pure state {}: \n",rho_bell_phi_plus);
  // let rho_bell_2 = rho_bell_phi_plus.clone();
  // let rho_bell_sqrd = find_dens_matrix_sqrd(rho_bell_2);
  // println!("Dens matrix squared is \n {} \n",rho_bell_sqrd);
  // println!("The purity is {} \n", find_purity(rho_bell_sqrd.clone()));
  // println!("The purity ranges from {} to {}", 1./find_dim(rho_bell_sqrd.clone()), 1);

//////////////////////////////////////////////////////////////////////////  
  //Fidelity test

  // println!("The fidelity for a maximally mixed state with itself is {} \n",fidelity(rho_max_mixed_2_qbit.clone(),rho_max_mixed_2_qbit.clone()));
  // println!("The fidelity for a maximally mixed state with phi_+ is {} \n",fidelity(rho_max_mixed_2_qbit.clone(),rho_bell_phi_plus.clone()));
  // println!("The fidelity for a maximally mixed state with phi_- is {} \n",fidelity(rho_max_mixed_2_qbit.clone(),rho_bell_phi_minus.clone()));  
  // println!("The fidelity for a maximally mixed state with psi_+  is {} \n",fidelity(rho_max_mixed_2_qbit.clone(),rho_bell_psi_plus.clone()));  
  // println!("The fidelity for a maximally mixed state with psi_-  is {} \n",fidelity(rho_max_mixed_2_qbit.clone(),rho_bell_psi_minus.clone()));  

  // println!("The fidelity for psi_- with psi_-  is {} \n",fidelity(rho_bell_psi_minus.clone(),rho_bell_psi_minus.clone()));
  // println!("The fidelity for psi_+ with psi_-  is {} \n",fidelity(rho_bell_psi_plus.clone(),rho_bell_psi_minus.clone()));   
  // println!("Fidelity of cos({:.4})|00> + sin({:.4})|11> is {}", theta, theta, fidelity_test(rho_part_entangled.clone(), rho_part_entangled.clone()));   

//////////////////////////////////////////////////////////////////////////  
  //Concurrence Tests
  
  // println!("Concurrence of phi_+ is {}", find_concurrence(rho_bell_phi_plus));
  // println!("Concurrence of phi_- is {}", find_concurrence(rho_bell_phi_minus));
  // println!("Concurrence of psi_+ is {}", find_concurrence(rho_bell_psi_plus));
  // println!("Concurrence of psi_- is {}", find_concurrence(rho_bell_psi_minus));

  // println!("Concurrence of mixed state is {}", find_concurrence(rho_max_mixed_2_qbit));
  // println!("Concurrence of cos({:.4})|00> + sin({:.4})|11> is {}", theta, theta,find_concurrence(rho_part_entangled));  

////////////////////////////////////////////////////////////////////////////
  //partial transpose test

  println!("{}", find_partial_transpose(test_mat))

}
