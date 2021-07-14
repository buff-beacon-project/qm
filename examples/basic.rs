use ndarray_linalg::*;
use std::f64::consts::PI;
use qm::*;
use ndarray::prelude::*;
// use std::fs::File;
// use std::io::{self, prelude::*, BufReader};
extern crate lazy_static;
use lazy_static::lazy_static;

  pub const NORM_CONST: f64 = 1./std::f64::consts::SQRT_2;
  pub const THETA: f64 = PI*(30./180.);

lazy_static!{
  pub static ref BELL_PHI_PLUS_COEF: VecC64 = array![c64::new(NORM_CONST , 0.0) , c64::new(0.0        , 0.0) , 
                                                     c64::new(0.0        , 0.0) , c64::new(NORM_CONST , 0.0) ];

  pub static ref BELL_PHI_MINUS_COEF: VecC64 = array![c64::new(NORM_CONST , 0.0) , c64::new(0.0         , 0.0) , 
                                                      c64::new(0.0        , 0.0) , c64::new(-NORM_CONST , 0.0) ];

  pub static ref BELL_PSI_PLUS_COEF: VecC64 = array![c64::new(0.0        , 0.0) , c64::new(NORM_CONST , 0.0) , 
                                                     c64::new(NORM_CONST , 0.0) , c64::new(0.0        , 0.0) ];

  pub static ref BELL_PSI_MINUS_COEF: VecC64 = array![c64::new(0.0         , 0.0) , c64::new(NORM_CONST , 0.0) , 
                                                      c64::new(-NORM_CONST , 0.0) , c64::new(0.0 ,        0.0) ];

  pub static ref RHO_MAX_MIXED_1_QBIT: MatrixC64 = array![ [c64::new(0.5 , 0.0) , c64::new(0.0 , 0.0)] ,
                                                           [c64::new(0.0 , 0.0) , c64::new(0.5 , 0.0)] ];

  pub static ref RHO_MAX_MIXED_2_QBIT: MatrixC64 = array![ [c64::new(0.25 , 0.0) ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                                           [c64::new(0.0 , 0.0)  ,  c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                                           [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0) ] ,
                                                           [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0)] ];



  pub static ref PSI_PART_ENTANGLED: VecC64 = array![c64::new(THETA.cos(), 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(THETA.sin(), 0.0)];


}

pub fn main() {

  // test_dens_matrix();
  // test_purity();
  // test_fidelity();
  // test_concurrence();
  test_partial_transpose()
}

pub fn test_dens_matrix() {

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_COEF);
  let rho_bell_phi_minus = create_dens_matrix(&BELL_PHI_MINUS_COEF);
  let rho_bell_psi_plus = create_dens_matrix(&BELL_PSI_PLUS_COEF);
  let rho_bell_psi_minus = create_dens_matrix(&BELL_PSI_MINUS_COEF);

  println!("dens matrix for phi_+ = \n {} \n", rho_bell_phi_plus);
  println!("dens matrix for phi_- = \n {} \n", rho_bell_phi_minus);
  println!("dens matrix for psi_+ = \n {} \n", rho_bell_psi_plus);
  println!("dens matrix for psi_- = \n {} \n", rho_bell_psi_minus);
}

pub fn test_purity() {

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_COEF);

  let rho_sqrd_1_qbit = find_matrix_sqrd(RHO_MAX_MIXED_1_QBIT.clone());
  println!("Dens matrix squared is \n {} \n",rho_sqrd_1_qbit);
  println!("The purity is {} \n", find_purity(rho_sqrd_1_qbit.clone()));
  println!("The purity ranges from {} to {} \n", 1./(find_dim(rho_sqrd_1_qbit.clone()) as f64), 1);

  println!("Dens matrix for a pure state {}: \n",rho_bell_phi_plus);
  let rho_bell_2 = rho_bell_phi_plus.clone();
  let rho_bell_sqrd = find_matrix_sqrd(rho_bell_2);
  println!("Dens matrix squared is \n {} \n",rho_bell_sqrd);
  println!("The purity is {} \n", find_purity(rho_bell_sqrd.clone()));
  println!("The purity ranges from {} to {}", 1./(find_dim(rho_bell_sqrd.clone()) as f64), 1);
}

pub fn test_fidelity() {

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_COEF);
  let rho_bell_phi_minus = create_dens_matrix(&BELL_PHI_MINUS_COEF);
  let rho_bell_psi_plus = create_dens_matrix(&BELL_PSI_PLUS_COEF);
  let rho_bell_psi_minus = create_dens_matrix(&BELL_PSI_MINUS_COEF);
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

  let rho_max_mixed_2_qbit: MatrixC64 = array![ [c64::new(0.25 , 0.0) ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                                           [c64::new(0.0 , 0.0)  ,  c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0) ] ,
                                                           [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0) , c64::new(0.0 , 0.0) ] ,
                                                           [c64::new(0.0 , 0.0)  ,  c64::new(0.0 , 0.0)  , c64::new(0.0 , 0.0)  , c64::new(0.25 , 0.0)] ];

  let rho_bell_phi_plus = create_dens_matrix(&BELL_PHI_PLUS_COEF);
  let rho_bell_phi_minus = create_dens_matrix(&BELL_PHI_MINUS_COEF);
  let rho_bell_psi_plus = create_dens_matrix(&BELL_PSI_PLUS_COEF);
  let rho_bell_psi_minus = create_dens_matrix(&BELL_PSI_MINUS_COEF);
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
                                               [c64::new(13.0 , 0.0) , c64::new(14.0 , 0.0) , c64::new(15.0 , 0.0) , c64::new(16.0 , 0.0)] ];

  println!("{}", find_partial_transpose(test_matrix))
}

pub fn test_find_schmidt_number (){

let jsi = array![ [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 7.021809714456165265e-16, 4.021293951545408608e-09, 1.570136766519117454e-06, 1.041760284915982798e-01, 9.080741470819553296e-01],
[0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 2.309611857490753894e-15, 1.773545881548452692e-08, 1.598283086425751902e-05, 2.088473576164723289e-01, 7.918718236672437349e-01, 5.104223610049659840e-03],
[0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 3.813076177648403989e-15, 4.154878288694038212e-08, 5.874864414436735594e-05, 3.166752640019571130e-01, 6.396393923139815918e-01, 1.430658656481265849e-03, 2.591396799672607065e-06],
[0.000000000000000000e+00, 0.000000000000000000e+00, 3.528727031220724971e-15, 5.502756374660440428e-08, 1.097525655936956328e-04, 3.946537567897805410e-01, 5.251928563793358817e-01, 4.952501274448888741e-04, 7.240749233562507969e-07, 1.639268140401811614e-13],
[0.000000000000000000e+00, 1.880116075542798807e-15, 4.214406414907862319e-08, 1.188279232498717962e-04, 4.248291793840358044e-01, 4.664077279677628640e-01, 2.380524956184089066e-04, 2.924975908383547222e-07, 9.433660289775034034e-14, 8.233482659501227230e-19],
[5.344874468905552523e-16, 1.835504869689886867e-08, 7.619227019601224353e-05, 4.018687767926586996e-01, 4.623394937691299722e-01, 1.744791231608409086e-04, 1.825092865050805734e-07, 5.981871840305573926e-14, 1.560295586677734415e-19, 0.000000000000000000e+00],
[4.297388239529948208e-09, 2.708028378190520979e-05, 3.291201868739798697e-01, 5.120927384882910660e-01, 2.046250527243471593e-04, 1.819631699331284768e-07, 5.312868277849745074e-14, 6.628570309055803741e-20, 0.000000000000000000e+00, 0.000000000000000000e+00],
[4.406383104956292129e-06, 2.226112983515576571e-01, 6.153364928513979093e-01, 3.774586386540264770e-04, 2.905589867532195520e-07, 7.211894379941240009e-14, 6.528074470650047608e-20, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00],
[1.147885806404641829e-01, 7.552940995327456219e-01, 1.010696671747382275e-03, 7.217726740643806409e-07, 1.469577112565755087e-13, 1.495080562659172714e-19, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00],
[8.645318651653516318e-01, 3.478721715992792122e-03, 2.624454533516879149e-06, 3.952731892264990721e-13, 7.744106501510905693e-19, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00]
];

println!("{}",find_schmidt_number(jsi));

}
  // println!("In file {}", "JSI_data.txt");
  //   let contents = fs::read_to_string("JSI_data.txt")
  //       .expect("Something went wrong reading the file");

  //   println!("With text:\n{:?}", contents);



