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

pub fn main() {

	let signal = read_f64_vector("signal_data200.csv".to_string()).unwrap();
	let idler = read_f64_vector("idler_data200.csv".to_string()).unwrap();
	let jsa = read_c64_array("jsa_data200.csv".to_string(),200).unwrap();
	let dt = 1.0e+06_f64;
	let norm = find_two_source_hom(signal, idler, jsa, dt);
	dbg!(norm);
}


pub fn read_c64_array(csv_file: String, array_size: usize) -> Result<MatrixC64, Box<dyn Error>> {

    let file = File::open(csv_file)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: MatrixC64 = reader.deserialize_array2((array_size, array_size))?;
    Ok(array_read)
}

pub fn read_f64_vector(csv_file: String) -> Result<VecF64, Box<dyn Error>> {

    let file = File::open(csv_file)?;
    let mut reader = ReaderBuilder::new()
    	.from_reader(file);
    	// .has_headers(false);

    let v : Vec<f64> = reader.headers()?
    	.into_iter()
    	.map(|s| s.parse::<f64>().unwrap()) // to f64
    	.collect(); // create a vec
	
	let arr = Array::from(v);
    Ok(arr)
}