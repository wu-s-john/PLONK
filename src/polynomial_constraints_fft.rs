use crate::execution_trace_polynomial::ExecutionTrace;
use ark_ff::{FftField, Field};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, UVPolynomial};

/// Builds the permutation constraint polynomial for PLONK's permutation argument using FFT operations.
///
/// This function takes an ExecutionTrace with polynomials in evaluation form and computes
/// the permutation constraint polynomial.
pub fn build_permutation_constraint_polynomial_fft<F: FftField>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    z_shifted: &Vec<F>,
    domain: &impl EvaluationDomain<F>,
    random1: F,
    random2: F,
) -> DensePolynomial<F> {
    let domain_size = domain.size();

    // Compute left side terms (original wire positions)
    // Left side: (a(X) + β*id1(X) + γ)(b(X) + β*id2(X) + γ)(c(X) + β*id3(X) + γ)(d(X) + β*id4(X) + γ)z(X)
    let non_permutated_terms = [
        // a(X) + β*id1(X) + γ
        execution_trace.input1.iter()
            .zip(&execution_trace.identity_pos1)
            .map(|(a, id)| *a + random1 * *id + random2)
            .collect::<Vec<F>>(),
        // b(X) + β*id2(X) + γ
        execution_trace.input2.iter()
            .zip(&execution_trace.identity_pos2)
            .map(|(b, id)| *b + random1 * *id + random2)
            .collect::<Vec<F>>(),
        // c(X) + β*id3(X) + γ
        execution_trace.input3.iter()
            .zip(&execution_trace.identity_pos3)
            .map(|(c, id)| *c + random1 * *id + random2)
            .collect::<Vec<F>>(),
        // d(X) + β*id4(X) + γ
        execution_trace.output.iter()
            .zip(&execution_trace.identity_out)
            .map(|(d, id)| *d + random1 * *id + random2)
            .collect::<Vec<F>>(),
    ];

    // Compute right side terms (permuted positions)
    // Right side: (a(X) + βSσ₁(X) + γ)(b(X) + βSσ₂(X) + γ)(c(X) + βSσ₃(X) + γ)(d(X) + βSσ₄(X) + γ)z(Xω)
    let permuted_terms = [
        // a(X) + βSσ₁(X) + γ
        execution_trace.input1.iter()
            .zip(&execution_trace.permutation_input1)
            .map(|(a, sigma)| *a + random1 * *sigma + random2)
            .collect::<Vec<F>>(),
        // b(X) + βSσ₂(X) + γ
        execution_trace.input2.iter()
            .zip(&execution_trace.permutation_input2)
            .map(|(b, sigma)| *b + random1 * *sigma + random2)
            .collect::<Vec<F>>(),
        // c(X) + βSσ₃(X) + γ
        execution_trace.input3.iter()
            .zip(&execution_trace.permutation_input3)
            .map(|(c, sigma)| *c + random1 * *sigma + random2)
            .collect::<Vec<F>>(),
        // d(X) + βSσ₄(X) + γ
        execution_trace.output.iter()
            .zip(&execution_trace.permutation_output)
            .map(|(d, sigma)| *d + random1 * *sigma + random2)
            .collect::<Vec<F>>(),
    ];

    // Calculate left and right products including all 4 terms
    let mut left_product = execution_trace.grand_product.clone();
    for term in &non_permutated_terms {
        left_product = left_product.iter()
            .zip(term)
            .map(|(acc, term_val)| *acc * *term_val)
            .collect();
    }

    let mut right_product = z_shifted.clone();
    for term in &permuted_terms {
        right_product = right_product.iter()
            .zip(term)
            .map(|(acc, term_val)| *acc * *term_val)
            .collect();
    }

    // Final constraint: left_product - right_product
    let constraint_evals: Vec<F> = left_product
        .iter()
        .zip(&right_product)
        .map(|(l, r)| *l - *r)
        .collect();

    DensePolynomial::from_coefficients_vec(domain.ifft(&constraint_evals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution_trace::{expand_execution_trace_table, parse_to_execution_trace},
        execution_trace_polynomial::{build_execution_trace_extended_evaluation_form, build_execution_trace_polynomial},
        grand_product::build_grand_product_evaluation_vector,
    };
    use ark_bn254::Fr as F;
    use ark_ff::{UniformRand, Zero};
    use ark_poly::{univariate::DensePolynomial, Polynomial, Radix2EvaluationDomain};
    use ark_std::test_rng;

    // This is a helper function, not a test
    fn test_build_permutation_constraint_polynomial_fft_impl(input: &str) {
        let execution_trace_table = parse_to_execution_trace(input);

        // Generate random challenges
        let random1 = F::rand(&mut test_rng());
        let random2 = F::rand(&mut test_rng());

        // Create coset domain (base size)
        let coset_domain_size = execution_trace_table.input1.len().next_power_of_two();
        let expanded_execution_trace_table =
            expand_execution_trace_table(&execution_trace_table, coset_domain_size);
        let coset_domain = Radix2EvaluationDomain::<F>::new(coset_domain_size)
            .expect("Failed to create coset domain");

        // Create evaluation domain (4x coset size)
        let evaluation_domain_size = coset_domain_size * 8;
        let evaluation_domain = Radix2EvaluationDomain::<F>::new(evaluation_domain_size)
            .expect("Failed to create evaluation domain");

        // Build grand product vector
        let grand_product_vector = build_grand_product_evaluation_vector(
            &expanded_execution_trace_table,
            Some(random1),
            Some(random2),
        );

        // Build polynomial form using coset domain
        let trace_poly: ExecutionTrace<F, DensePolynomial<F>> = build_execution_trace_polynomial(
            &expanded_execution_trace_table,
            grand_product_vector,
            random1,
            random2,
            &coset_domain,
        );

        // Convert to evaluation form over evaluation domain
        let extended_trace_eval = build_execution_trace_extended_evaluation_form(
            &trace_poly,
            &coset_domain,
            &evaluation_domain,
        );

        // Compute the shifted grand product polynomial z(ωX)
        // First, get z(X) evaluations
        let z_evals = &extended_trace_eval.grand_product;

        // Create z(ωX) evaluations by shifting indices
        let n = z_evals.len();
        let mut z_shifted = vec![F::zero(); n];

        // Shift by one position in the evaluation domain (equivalent to multiplying by ω)
        // This is because evaluation_domain.element(i+1) = evaluation_domain.element(i) * ω
        for i in 0..n - 1 {
            z_shifted[i] = z_evals[i + 1];
        }
        // Handle the wrap-around case
        z_shifted[n - 1] = z_evals[0];

        // Build constraint polynomial using the FFT-based method
        let constraint_poly = build_permutation_constraint_polynomial_fft(
            &extended_trace_eval,
            &z_shifted,
            &evaluation_domain,
            random1,
            random2,
        );

        // Evaluate at all points in the evaluation domain
        let constraint_evals = coset_domain
            .elements()
            .map(|x| constraint_poly.evaluate(&x))
            .collect::<Vec<_>>();

        // Print first few evaluations for debugging
        println!(
            "First 5 constraint evaluations: {:?}",
            &constraint_evals[0..5.min(constraint_evals.len())]
        );

        // Validation checks
        for (i, eval) in constraint_evals.iter().enumerate() {
            assert!(
                eval.is_zero(),
                "Constraint polynomial should evaluate to 0 at domain element {}, got: {}",
                i,
                eval
            );
        }
    }

    // Actual test functions that call the helper function
    #[test]
    #[ignore]
    fn test_build_permutation_constraint_polynomial_let_x_5_in_x_plus_x_times_x_fft() {
        test_build_permutation_constraint_polynomial_fft_impl("let x = 5 in x + x * x");
    }

    #[test]
    #[ignore]
    fn test_build_permutation_constraint_polynomial_boolean_logic_fft() {
        test_build_permutation_constraint_polynomial_fft_impl(
            "let a = true in 
             let b = false in 
             if a && b then 1 else 0",
        );
    }

    #[test]
    #[ignore]
    fn test_build_permutation_constraint_polynomial_function_composition_fft() {
        test_build_permutation_constraint_polynomial_fft_impl(
            "let compose = fun f -> fun g -> fun x -> f (g x) in
            let add1 = fun x -> x + 1 in
            let mul2 = fun x -> x * 2 in
            compose add1 mul2 5",
        );
    }
    
    #[test]
    #[ignore]
    fn test_build_permutation_constraint_polynomial_simple_fft() {
        test_build_permutation_constraint_polynomial_fft_impl("1+2");
    }

    #[test]
    #[ignore]
    fn test_build_permutation_constraint_polynomial_simple_2_fft() {
        test_build_permutation_constraint_polynomial_fft_impl("1+2*3");
    }
} 