use ark_ff::{Field, One, Zero};
use ark_poly::Polynomial;
use ark_std::rand::thread_rng;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

use crate::kzg::group_projectable::GroupProjectable;


// Custom pairing trait that gives more flexibility than the default one
trait PairingOp<G1, G2> {
    type Output;

    fn pairing(left: G1, right: G2) -> Self::Output;
}

/// Verifier for batch KZG proofs
pub struct BatchKZGVerifier<F, P, G1Proj, G2Proj, G1, G2, E>
where
    F: Field,
    P: Polynomial<F>,
    G1Proj: GroupProjectable<F, P, Commitment = G1>,
    G2Proj: GroupProjectable<F, P, Commitment = G2>,
    G1: Clone,
    G2: Clone,
    E: PairingOp<G1, G2>,
{
    /// Projector for G1 commitments
    pub g1_projector: G1Proj,
    /// Projector for G2 commitments
    pub g2_projector: G2Proj,
    _marker: PhantomData<(F, P, E)>,
}

impl<F, P, G1Proj, G2Proj, G1, G2, E> BatchKZGVerifier<F, P, G1Proj, G2Proj, G1, G2, E>
where
    F: Field,
    P: Polynomial<F>,
    G1Proj: GroupProjectable<F, P, Commitment = G1>,
    G2Proj: GroupProjectable<F, P, Commitment = G2>,
    G1: Clone + Add<Output = G1> + Sub<Output = G1> + Mul<F, Output = G1> + Zero,
    G2: Clone + Sub<Output = G2> + Mul<F, Output = G2>,
    E: PairingOp<G1, G2>,
    E::Output: Eq + One,
{
    /// Creates a new verifier
    pub fn new(g1_projector: G1Proj, g2_projector: G2Proj) -> Self {
        Self {
            g1_projector,
            g2_projector,
            _marker: PhantomData,
        }
    }

    /// Samples random evaluation points for batch verification
    ///
    /// # Returns
    /// Two random evaluation points (z, z')
    pub fn sample_evaluation_points(&self) -> (F, F) {
        let mut rng = ark_std::rand::thread_rng();
        (F::rand(&mut rng), F::rand(&mut rng))
    }

    /// Samples random scalars for linear combinations in batch verification
    ///
    /// # Returns
    /// Two random scalars (γ, γ') for linear combinations
    pub fn sample_for_linear_combination(&self) -> (F, F) {
        let mut rng = ark_std::rand::thread_rng();
        (F::rand(&mut rng), F::rand(&mut rng))
    }

    /// Samples a random scalar for combining proofs
    ///
    /// # Returns
    /// A random scalar r'
    pub fn sample_r_prime(&self) -> F {
        let mut rng = thread_rng();
        F::rand(&mut rng)
    }

    /// Computes the F term for batch verification
    ///
    /// # Arguments
    /// * `commitment1s` - Vector of commitments to polynomials evaluated at point1
    /// * `p1s` - Evaluations of polynomials at point1
    /// * `gamma1` - First random scalar for linear combination
    /// * `commitment2s` - Vector of commitments to polynomials evaluated at point2
    /// * `p2s` - Evaluations of polynomials at point2
    /// * `gamma2` - Second random scalar for linear combination
    /// * `r_prime` - Random scalar for combining proofs
    ///
    /// # Returns
    /// The F term for the verification equation
    pub fn compute_f_term(
        &self,
        commitment1s: &[G1],
        p1s: &[F],
        gamma1: &F,
        commitment2s: &[G1],
        p2s: &[F],
        gamma2: &F,
        r_prime: &F,
    ) -> G1 {
        // Compute the first term:
        // (∑(i=1 to t1)[γ^(i−1) * cmi] − [∑(i=1 to t1)[γ^(i−1) * si]]1)
        let mut sum_commits1 = G1::zero();
        let mut sum_evals1 = F::zero();
        
        for (i, (commit, eval)) in commitment1s.iter().zip(p1s.iter()).enumerate() {
            let gamma_power = gamma1.pow([i as u64]);
            
            // Add γ^(i-1) * commit to sum_commits1
            sum_commits1 = sum_commits1 + commit.clone().mul(gamma_power);
            
            // Add γ^(i-1) * eval to sum_evals1
            sum_evals1 += gamma_power * eval;
        }
        
        // Compute [∑(i=1 to t1)[γ^(i−1) * si]]1
        let sum_evals1_committed = self.g1_projector.commit_scalar(sum_evals1);
        
        // Compute (∑(i=1 to t1)[γ^(i−1) * cmi] − [∑(i=1 to t1)[γ^(i−1) * si]]1)
        let first_term = sum_commits1 - sum_evals1_committed;

        // Compute the second term:
        // (∑(i=1 to t2)[γ'^(i−1) * cm'i] − [∑(i=1 to t2)[γ'^(i−1) * s'i]]1)
        let mut sum_commits2 = G1::zero();
        let mut sum_evals2 = F::zero();
        
        for (i, (commit, eval)) in commitment2s.iter().zip(p2s.iter()).enumerate() {
            let gamma_power = gamma2.pow([i as u64]);
            
            // Add γ'^(i-1) * commit to sum_commits2
            sum_commits2 = sum_commits2 + commit.clone().mul(gamma_power);
            
            // Add γ'^(i-1) * eval to sum_evals2
            sum_evals2 += gamma_power * eval;
        }
        
        // Compute [∑(i=1 to t2)[γ'^(i−1) * s'i]]1
        let sum_evals2_committed = self.g1_projector.commit_scalar(sum_evals2);
        
        // Compute (∑(i=1 to t2)[γ'^(i−1) * cm'i] − [∑(i=1 to t2)[γ'^(i−1) * s'i]]1)
        let second_term = sum_commits2 - sum_evals2_committed;
        
        // Compute F = first_term + r' * second_term
        first_term + second_term.mul(*r_prime)
    }

    /// Verifies a batch KZG proof
    ///
    /// # Arguments
    /// * `commitment1s` - Vector of commitments to polynomials evaluated at point1
    /// * `p1s` - Evaluations of polynomials at point1
    /// * `point1` - First evaluation point (z)
    /// * `commitment2s` - Vector of commitments to polynomials evaluated at point2
    /// * `p2s` - Evaluations of polynomials at point2
    /// * `point2` - Second evaluation point (z')
    /// * `w` - First witness commitment (W)
    /// * `w_prime` - Second witness commitment (W')
    /// * `r_prime` - Random scalar for combining proofs
    ///
    /// # Returns
    /// True if the proof is valid, false otherwise
    pub fn verify_batch_kzg(
        &self,
        commitment1s: &[G1],
        p1s: &[F],
        point1: &F,
        commitment2s: &[G1],
        p2s: &[F],
        point2: &F,
        w: &G1,
        w_prime: &G1,
        r_prime: &F,
    ) -> bool {
        // Sample random gammas if they weren't provided
        let (gamma1, gamma2) = self.sample_for_linear_combination();
        
        // Compute the F term
        let f = self.compute_f_term(
            commitment1s,
            p1s,
            &gamma1,
            commitment2s,
            p2s,
            &gamma2,
            r_prime,
        );
        
        // Compute the left side of the verification equation:
        // F + z·W + r′·z′·W′
        let left_g1 = f + w.clone().mul(*point1) + w_prime.clone().mul(*point2 * *r_prime);
        
        // Compute the right side:
        // −W − r′·W′
        let right_g1 = G1::zero() - w.clone() - w_prime.clone().mul(*r_prime);
        
        // Get g2 generator (equivalent to [1]_2)
        let g2 = self.g2_projector.commit_scalar(F::one());
        
        // Get g2_s (equivalent to [s]_2 where s is the trusted setup secret)
        // For this, we would need a specific way to access it from the G2Proj
        // For now, assuming G2Proj has a method to get this value
        let g2_s = self.g2_projector.commit_scalar(F::from(2u64)); // Placeholder, would need proper implementation
        
        // Compute g2_s - point1 * g2 (equivalent to [s-z]_2)
        let g2_s_minus_z = g2_s - g2.clone().mul(*point1);
        
        // Compute pairings and check if e(left_g1, g2) * e(right_g1, g2_s_minus_z) = 1
        let pairing1 = E::pairing(left_g1.into(), g2.into());
        let pairing2 = E::pairing(right_g1.into(), g2_s_minus_z.into());
        
        // The final check: e(left_g1, g2) * e(right_g1, g2_s_minus_z) = 1
        // Which is equivalent to: pairing1 * pairing2 == E::Output::one()
        pairing1 * pairing2 == E::Output::one()
    }
}