use ark_ff::Field; // Example import
use ark_std::{test_rng, UniformRand};
use ark_tweedle::Fr;
mod ast;
mod execution_trace;
mod offset_table;
mod permutation_polynomial;
mod polynomial_utils;
use ark_poly_commit::{kzg10, PolynomialCommitment};


fn main() {
    let mut rng = test_rng();
    let random_field_element = <Fr as UniformRand>::rand(&mut rng);
    println!("Random field element: {:?}", random_field_element);
}