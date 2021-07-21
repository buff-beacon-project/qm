// use std::f64::consts::PI;
use qm::*;
extern crate ndarray;
use ndarray::prelude::*;

extern crate csv;
extern crate ndarray_csv;

use csv::{ReaderBuilder};
use ndarray_csv::{Array2Reader};
use std::error::Error;
use std::fs::File;


//TODO: Original output of jsa -> convert from (re + im*j) to c64::new(re, im)

pub fn main(){

	let signal = array![1.444682468414305251e-06, 1.497341234207152617e-06, 1.549999999999999983e-06, 1.602658765792847349e-06, 1.655317531585694715e-06];
	let idler = array![1.457282220278072330e-06, 1.510931735289744386e-06, 1.564581250301416654e-06, 1.618230765313088710e-06, 1.671880280324760978e-06];
	let jsa = read_c64_array("JSA_data5.csv".to_string(),5).unwrap();
	let dt = 1.0e+06_f64;
	let norm = find_two_source_hom(signal, idler, jsa,dt);
	println!("{:?}",norm);
}


pub fn read_c64_array(csv_file: String, array_size: usize) -> Result<MatrixC64, Box<dyn Error>> {

    let file = File::open(csv_file)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: MatrixC64 = reader.deserialize_array2((array_size, array_size))?;
    println!("{}",array_read);
    Ok(array_read)
}