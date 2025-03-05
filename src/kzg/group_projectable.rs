use ark_ff::Field;
use ark_poly::{DenseUVPolynomial, Polynomial};
use ark_ec::Group;
use ark_poly::univariate::DensePolynomial;

/// A trait for converting polynomials and scalar values into commitment objects.
///
/// This trait provides methods to commit to both polynomials and scalar values,
/// producing a commitment object that can be used in cryptographic protocols.
pub trait GroupProjectable<F: Field, P: Polynomial<F>> {
    /// The type of commitment produced
    type Commitment;
    
    /// Commits to a polynomial
    fn commit(&self, polynomial: &P) -> Self::Commitment;
    
    /// Commits to a scalar value
    fn commit_scalar(&self, point: F) -> Self::Commitment;
}

pub struct FieldCommitment<F: Field> {
    pub sample_point: F
}

pub struct SRS<G: Group> {
    pub g1_powers: Vec<G>,
}

// Simpler implementation for Field type directly
impl<F: Field> GroupProjectable<F, DensePolynomial<F>> for FieldCommitment<F> {
    type Commitment = F;
    
    fn commit(&self, polynomial: &DensePolynomial<F>) -> Self::Commitment {
        polynomial.evaluate(&self.sample_point)
    }
    
    fn commit_scalar(&self, point: F) -> Self::Commitment {
        point
    }
}

impl<F: Field, G: Group<ScalarField = F>> GroupProjectable<F, DensePolynomial<F>> for SRS<G> {
    type Commitment = G;
    
    /// Creates a KZG commitment to a polynomial using the SRS
    /// 
    /// C = ∑(i=0 to d) coeff_i * g1^(s^i)
    fn commit(&self, polynomial: &DensePolynomial<F>) -> Self::Commitment {
        let coeffs = polynomial.coeffs();
        
        // If the polynomial degree exceeds our SRS size, we can't compute the commitment
        assert!(
            coeffs.len() <= self.g1_powers.len(),
            "Polynomial degree too large for this SRS"
        );
        
        // Compute the linear combination: ∑ coeff_i * g1_powers[i]
        coeffs
            .iter()
            .zip(self.g1_powers.iter())
            .map(|(coeff, g1_power)| g1_power.mul(*coeff))
            .sum()
    }
    
    fn commit_scalar(&self, point: F) -> Self::Commitment {
        // Ensure the SRS has at least one element
        assert!(!self.g1_powers.is_empty(), "SRS is empty");
        
        // Multiply the first element of the SRS by the scalar
        self.g1_powers[0].mul(point)
    }
}