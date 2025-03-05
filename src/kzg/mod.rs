pub mod prover;
pub mod verifier;
use std::marker::PhantomData;
pub mod group_projectable;

use ark_ec::{CurveGroup, pairing::Pairing};
use ark_ff::{Field, Zero};
use ark_poly::{univariate::DensePolynomial, Polynomial, DenseUVPolynomial};


/// Represents a batch of KZG commitments to polynomials.
#[derive(Debug, Clone)]
pub struct BatchKZGCommitment<F: Field, G: CurveGroup<ScalarField = F>> {
    pub commitments: Vec<G>,
}

/// Represents a batch proof for multiple polynomials evaluated at multiple points.
#[derive(Debug, Clone)]
pub struct BatchKZGProof<F: Field, G: CurveGroup<ScalarField = F>> {
    pub first_point_witness: G,  // W in the protocol
    pub second_point_witness: G, // W' in the protocol
    pub combined_term: G,        // F in the protocol
    pub eval_points: Vec<F>,     // z, z' in the protocol
    pub evaluations: Vec<Vec<F>>, // evaluations at each point for each polynomial
}

/// A trait representing a challenge generator for the verifier
pub trait VerifierChallenge<F: Field> {
    fn generate_challenge(&self, input: &[u8]) -> F;
}

// Extension of KZGSystem trait to support batch operations
pub trait BatchKZGSystem<F: Field, G1: CurveGroup<ScalarField = F>, G2: CurveGroup<ScalarField = F>> {
    type E: Pairing;
    type Poly: DenseUVPolynomial<F> + Clone;

    fn batch_prove(
        &self, 
        polynomials: &[Self::Poly], 
        eval_points: &[F]
    ) -> BatchKZGProof<F, G1>
    where
        <Self::E as Pairing>::ScalarField: From<F>,
        <Self::E as Pairing>::G1Affine: From<G1>,
        <Self::E as Pairing>::G2Affine: From<G2>;

    fn batch_verify(&self, proof: BatchKZGProof<F, G1>) -> bool
    where
        <Self::E as Pairing>::ScalarField: From<F>,
        <Self::E as Pairing>::G1Affine: From<G1>,
        <Self::E as Pairing>::G2Affine: From<G2>;
}