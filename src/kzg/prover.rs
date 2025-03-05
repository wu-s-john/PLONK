use ark_ff::Field;
use ark_poly::DenseUVPolynomial;
use std::{marker::PhantomData, ops::{Div, MulAssign}};

use crate::kzg::group_projectable::GroupProjectable;

/// Prover for batch KZG commitments and proofs
pub struct BatchKZGProver<
    F: Field,
    P: DenseUVPolynomial<F, Point = F>,
    Proj: GroupProjectable<F, P>,
    G
> {
    /// Group projectable implementation for KZG commitments
    pub projector: Proj,
    _marker: PhantomData<(F, P, G)>,
}

impl<F, P, Proj, G> BatchKZGProver<F, P, Proj, G>
where
    F: Field,
    P: DenseUVPolynomial<F> + Div<Output = P> + MulAssign<F> + Clone,
    Proj: GroupProjectable<F, P, Commitment = G>,
{
    /// Creates a new prover with the given projector
    pub fn new(projector: Proj) -> Self {
        Self {
            projector,
            _marker: PhantomData,
        }
    }

    /// Commits to a batch of polynomials.
    ///
    /// # Arguments
    /// * `polynomials` - Vector of polynomials to commit to
    ///
    /// # Returns
    /// Vector of commitments to the polynomials, in the same order
    pub fn batch_commitments(&self, polynomials: &[P]) -> Vec<G> {
        polynomials
            .iter()
            .map(|p| self.projector.commit(p))
            .collect()
    }

    /// Evaluates a batch of polynomials at a given point.
    ///
    /// # Arguments
    /// * `polynomials` - Vector of polynomials to evaluate
    /// * `point` - Point at which to evaluate the polynomials
    ///
    /// # Returns
    /// Vector of all evaluations in order
    pub fn batch_eval(&self, polynomials: &[P], point: &F) -> Vec<F> {
        polynomials
            .iter()
            .map(|p| p.evaluate(point))
            .collect()
    }

    /// Computes aggregate quotient polynomial commitments.
    ///
    /// # Arguments
    /// * `difference_poly1` - First set of difference polynomials
    /// * `point1` - First evaluation point
    /// * `difference_poly2` - Second set of difference polynomials
    /// * `point2` - Second evaluation point
    /// * `gamma1` - First random scalar
    /// * `gamma2` - Second random scalar
    ///
    /// # Returns
    /// Tuple of commitments (W, W') where W is commitment to h(X) and W' is commitment to h'(X)
    pub fn compute_aggregate_quotient_polynomial_commitments(
        &self,
        difference_poly1: &[P],
        point1: F,
        difference_poly2: &[P],
        point2: F,
        gamma1: F,
        gamma2: F,
    ) -> (G, G) {
        // Create divisor polynomials (X - point1) and (X - point2)
        let divisor1 = P::from_coefficients_vec(vec![-point1, F::one()]);
        let divisor2 = P::from_coefficients_vec(vec![-point2, F::one()]);
        
        // Compute quotient polynomials for the first set
        let mut combined_quotient1 = P::from_coefficients_vec(vec![]);
        for (i, diff_poly) in difference_poly1.iter().enumerate() {
            // Instead of diff_poly / divisor1, do:
            let quotient : P = diff_poly.clone().div(divisor1.clone());
            
            
            // Multiply by gamma1^i
            let gamma_power = gamma1.pow([i as u64]);

            
            // scale q by gamma_power
            let mut scaled_quotient = quotient;
            scaled_quotient.mul_assign(gamma_power);

            // Add to combined quotient
            // combined_quotient1 += scaled_quotient
            let mut new_combined = combined_quotient1.clone();
            new_combined += &scaled_quotient;
            combined_quotient1 = new_combined;
        }
        
        // Compute quotient polynomials for the second set
        let mut combined_quotient2 = P::from_coefficients_vec(vec![]);
        for (i, poly) in difference_poly2.iter().enumerate() {
            let quotient : P = poly.clone().div(divisor2.clone());
            
            // Multiply by gamma2^i
            let gamma_power = gamma2.pow([i as u64]);

            let mut scaled_quotient = quotient;
            scaled_quotient.mul_assign(gamma_power);

            let mut new_combined = combined_quotient2.clone();
            new_combined += &scaled_quotient;
            combined_quotient2 = new_combined;
        }
        
        // Commit to the combined quotient polynomials
        let commitment1 = self.projector.commit(&combined_quotient1);
        let commitment2 = self.projector.commit(&combined_quotient2);
        
        (commitment1, commitment2)
    }
    
    /// Computes the quotient polynomial: (p(X) - v) / (X - z)
    ///
    /// # Arguments
    /// * `polynomial` - The polynomial p(X)
    /// * `eval_point` - The point z where polynomial is evaluated
    /// * `eval_value` - The expected value v = p(z)
    ///
    /// # Returns
    /// The quotient polynomial
    pub fn compute_quotient_polynomial(
        polynomial: &P,
        eval_point: &F,
        eval_value: &F,
    ) -> P {
        // Construct polynomial representing -eval_value
        let constant_poly = P::from_coefficients_vec(vec![-*eval_value]);
        
        // difference = polynomial - constant_poly
        let mut difference = polynomial.clone();
        difference -= &constant_poly;
        
        // Create divisor polynomial (X - z)
        let divisor = P::from_coefficients_vec(vec![-*eval_point, F::one()]);
        
        // Instead of difference / divisor, do:
        let quotient : P = difference.div(divisor);
        // Remainder should be zero if indeed difference has (X - z) as a factor
        assert!(quotient.is_zero(), "Non-zero remainder in quotient polynomial computation");

        quotient
    }
}
