use crate::execution_trace_polynomial::ExecutionTrace;
use ark_ff::{FftField, Field};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, UVPolynomial};

/// Builds the permutation constraint polynomial for PLONK's permutation argument using direct polynomial operations.
///
/// This function works directly with polynomials rather than evaluation vectors
pub fn build_permutation_constraint_polynomial_direct<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    domain: &impl EvaluationDomain<F>,
    random1: F,
    random2: F,
) -> DensePolynomial<F> {
    // Get all polynomials from the execution trace
    let input1_poly = &execution_trace.input1;
    let input2_poly = &execution_trace.input2;
    let input3_poly = &execution_trace.input3;
    let output_poly = &execution_trace.output;

    // Identity polynomials (wire positions)
    let id1_poly = &execution_trace.identity_pos1;
    let id2_poly = &execution_trace.identity_pos2;
    let id3_poly = &execution_trace.identity_pos3;
    let out_id_poly = &execution_trace.identity_out;

    // Permutation polynomials
    let sigma1_poly = &execution_trace.permutation_input1;
    let sigma2_poly = &execution_trace.permutation_input2;
    let sigma3_poly = &execution_trace.permutation_input3;
    let sigma_out_poly = &execution_trace.permutation_output;

    // Grand product polynomial
    let z_poly = &execution_trace.grand_product;

    // Compute z(Xω) properly by creating a shifted evaluation vector
    let z_evals = domain.fft(&z_poly.coeffs());
    let n = z_evals.len();
    let mut z_shifted_evals = vec![F::zero(); n];

    // Correct way to compute z(ωX): rotate indices by 1
    // z_shifted_evals[i] = z_evals[(i+1) % n] = z(ω * domain.element(i))
    for i in 0..n {
        let i_plus_1 = (i + 1) % n;
        z_shifted_evals[i] = z_evals[i_plus_1];
    }

    // Convert back to coefficient form
    let z_shifted_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&z_shifted_evals));

    // Term 1: a(X) + β*id1(X) + γ
    let term1_left = (input1_poly + &(id1_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Term 2: b(X) + β*id2(X) + γ
    let term2_left = (input2_poly + &(id2_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Term 3: c(X) + β*id3(X) + γ
    let term3_left = (input3_poly + &(id3_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Term 4: d(X) + β*id4(X) + γ
    let term4_left = (output_poly + &(out_id_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Term 1: a(X) + βSσ₁(X) + γ
    let term1_right = (input1_poly + &(sigma1_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Term 2: b(X) + βSσ₂(X) + γ
    let term2_right = (input2_poly + &(sigma2_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Term 3: c(X) + βSσ₃(X) + γ
    let term3_right = (input3_poly + &(sigma3_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Term 4: d(X) + βSσ₄(X) + γ
    let term4_right = (output_poly + &(sigma_out_poly * random1))
        + DensePolynomial::from_coefficients_vec(vec![random2]);

    // Use z(X) on left and z(ωX) on right to match standard PLONK formula

    // Left side: product of non-permuted terms with z(X)
    let left_term = &(&(&(&term1_left * &term2_left) * &term3_left) * &term4_left);

    // Right side: product of permuted terms with z(ωX)
    let right_term = &(&(&(&term1_right * &term2_right) * &term3_right) * &term4_right);

    // Final constraint: z(X)⋅left_term - z(ωX)⋅right_term = 0
    let constraint = &(z_poly * left_term) - &(&z_shifted_poly * right_term);

    constraint
}

/// Builds the addition constraint polynomial: (input_1 + input_2 - output) * add_selector
fn build_addition_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,  // Unused for gate constraints
    _gamma: F, // Unused for gate constraints
) -> DensePolynomial<F> {
    let add_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Add).unwrap();
    
    // (input_1 + input_2 - output) * add_selector
    let constraint = &(&(&execution_trace.input1 + &execution_trace.input2) - &execution_trace.output) * add_selector;
    
    constraint.clone()
}

/// Builds the subtraction constraint polynomial: (input_1 - input_2 - output) * sub_selector
fn build_subtraction_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let sub_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Sub).unwrap();
    
    // (input_1 - input_2 - output) * sub_selector
    let constraint = &(&(&execution_trace.input1 - &execution_trace.input2) - &execution_trace.output) * sub_selector;
    
    constraint.clone()
}

/// Builds the multiplication constraint polynomial: (input_1 * input_2 - output) * mult_selector
fn build_multiplication_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let mult_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Mult).unwrap();
    
    // (input_1 * input_2 - output) * mult_selector
    let constraint = &(&(&execution_trace.input1 * &execution_trace.input2) - &execution_trace.output) * mult_selector;
    
    constraint.clone()
}

/// Builds the division constraint polynomial: (input_1 - input_2 * output) * div_selector
fn build_division_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let div_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Div).unwrap();
    
    // (input_1 - input_2 * output) * div_selector
    let constraint = &(&execution_trace.input1 - &(&execution_trace.input2 * &execution_trace.output)) * div_selector;
    
    constraint.clone()
}

/// Builds the equality constraint polynomial: (input_1 - input_2) * (1 - output) * eq_selector
fn build_equality_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let eq_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Eq).unwrap();
    
    // (input_1 - input_2) * (1 - output) * eq_selector
    let diff = &execution_trace.input1 - &execution_trace.input2;
    let one = DensePolynomial::from_coefficients_vec(vec![F::one()]);
    let one_minus_output = &one - &execution_trace.output;
    let constraint = &(&diff * &one_minus_output) * eq_selector;
    
    constraint.clone()
}

/// Builds the negation constraint polynomial: (input_1 - !output) * not_selector
/// where !output is implemented as 1 - output for field elements
fn build_negation_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let not_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Not).unwrap();
    
    // (input_1 - (1 - output)) * not_selector
    let one = DensePolynomial::from_coefficients_vec(vec![F::one()]);
    let not_output = &one - &execution_trace.output;
    let constraint = &(&execution_trace.input1 - &not_output) * not_selector;
    
    constraint.clone()
}

/// Builds the if constraint polynomial: (input_1 * input_2 + (1-input_1) * input_3 - output) * if_selector
fn build_if_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let if_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::If).unwrap();
    
    // (input_1 * input_2 + (1-input_1) * input_3 - output) * if_selector
    let one = DensePolynomial::from_coefficients_vec(vec![F::one()]);
    let one_minus_cond = &one - &execution_trace.input1;
    let then_term = &execution_trace.input1 * &execution_trace.input2;
    let else_term = &one_minus_cond * &execution_trace.input3;
    let constraint = &(&(&then_term + &else_term) - &execution_trace.output) * if_selector;
    
    constraint.clone()
}

/// Builds the integer constraint polynomial: (input_1 - output) * int_selector
fn build_int_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let int_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Int).unwrap();
    
    // (input_1 - output) * int_selector
    let constraint = &(&execution_trace.input1 - &execution_trace.output) * int_selector;
    
    constraint.clone()
}

/// Builds the boolean constraint polynomial: (input_1 - output) * bool_selector
fn build_bool_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    _domain: &impl EvaluationDomain<F>,
    _beta: F,
    _gamma: F,
) -> DensePolynomial<F> {
    let bool_selector = execution_trace.selectors.get(&crate::plonk_circuit::PlonkNodeKind::Bool).unwrap();
    
    // (input_1 - output) * bool_selector
    let constraint = &(&execution_trace.input1 - &execution_trace.output) * bool_selector;
    
    constraint.clone()
}

/// Builds the complete gate constraint polynomial by combining all individual gate constraints
pub fn build_gate_constraint_polynomial<F: FftField>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    domain: &impl EvaluationDomain<F>,
    beta: F,
    gamma: F,
) -> DensePolynomial<F> {
    // Build individual constraint polynomials
    let int_poly = build_int_constraint_polynomial(execution_trace, domain, beta, gamma);
    let bool_poly = build_bool_constraint_polynomial(execution_trace, domain, beta, gamma);
    let add_poly = build_addition_constraint_polynomial(execution_trace, domain, beta, gamma);
    let sub_poly = build_subtraction_constraint_polynomial(execution_trace, domain, beta, gamma);
    let mult_poly = build_multiplication_constraint_polynomial(execution_trace, domain, beta, gamma);
    let div_poly = build_division_constraint_polynomial(execution_trace, domain, beta, gamma);
    let eq_poly = build_equality_constraint_polynomial(execution_trace, domain, beta, gamma);
    let not_poly = build_negation_constraint_polynomial(execution_trace, domain, beta, gamma);
    let if_poly = build_if_constraint_polynomial(execution_trace, domain, beta, gamma);

    // Sum all constraint polynomials
    let mut sum_poly = int_poly;
    sum_poly = sum_poly + bool_poly;
    sum_poly = sum_poly + add_poly;
    sum_poly = sum_poly + sub_poly;
    sum_poly = sum_poly + mult_poly;
    sum_poly = sum_poly + div_poly;
    sum_poly = sum_poly + eq_poly;
    sum_poly = sum_poly + not_poly;
    sum_poly = sum_poly + if_poly;

    sum_poly
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution_trace::{expand_execution_trace_table, parse_to_execution_trace},
        execution_trace_polynomial::build_execution_trace_polynomial,
        grand_product::build_grand_product_evaluation_vector,
    };
    use ark_bn254::Fr as F;
    use ark_ff::{UniformRand, Zero};
    use ark_poly::{univariate::DensePolynomial, Polynomial};
    use ark_std::test_rng;

    fn test_build_permutation_constraint_polynomial_direct_impl(input: &str) {
        let execution_trace_table = parse_to_execution_trace(input);

        // Generate random challenges
        let random1 = F::rand(&mut test_rng());
        let random2 = F::rand(&mut test_rng());

        // Create coset domain (base size)
        let coset_domain_size = execution_trace_table.input1.len().next_power_of_two();
        let expanded_execution_trace_table =
            expand_execution_trace_table(&execution_trace_table, coset_domain_size);
        let coset_domain = ark_poly::Radix2EvaluationDomain::<F>::new(coset_domain_size)
            .expect("Failed to create coset domain");

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

        let constraint_poly = build_permutation_constraint_polynomial_direct(
            &trace_poly,
            &coset_domain,
            random1,
            random2,
        );

        // Evaluate at all points in the evaluation domain
        let constraint_evals = coset_domain
            .elements()
            .map(|x| constraint_poly.evaluate(&x))
            .collect::<Vec<_>>();
        
        // Validation checks
        for eval in constraint_evals {
            assert!(eval.is_zero(), "Constraint polynomial should evaluate to 0");
        }
    }

    // New test function for build_gate_constraint_polynomial
    fn test_build_gate_constraint_polynomial_impl(input: &str) {
        let execution_trace_table = parse_to_execution_trace(input);

        // Generate random challenges
        let random1 = F::rand(&mut test_rng());
        let random2 = F::rand(&mut test_rng());

        // Create coset domain (base size)
        let coset_domain_size = execution_trace_table.input1.len().next_power_of_two();
        let expanded_execution_trace_table =
            expand_execution_trace_table(&execution_trace_table, coset_domain_size);
        let coset_domain = ark_poly::Radix2EvaluationDomain::<F>::new(coset_domain_size)
            .expect("Failed to create coset domain");

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

        // Build gate constraint polynomial
        let constraint_poly = build_gate_constraint_polynomial(
            &trace_poly,
            &coset_domain,
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
            "First 5 gate constraint evaluations: {:?}",
            &constraint_evals[0..5.min(constraint_evals.len())]
        );
        
        // Validation checks
        for (i, eval) in constraint_evals.iter().enumerate() {
            assert!(
                eval.is_zero(),
                "Gate constraint polynomial should evaluate to 0 at domain element {}, got: {}",
                i,
                eval
            );
        }
    }

    // Tests for permutation constraint polynomial
    #[test]
    fn test_build_permutation_constraint_polynomial_let_x_5_in_x_plus_x_times_x() {
        test_build_permutation_constraint_polynomial_direct_impl("let x = 5 in x + x * x");
    }

    #[test]
    fn test_build_permutation_constraint_polynomial_boolean_logic() {
        test_build_permutation_constraint_polynomial_direct_impl(
            "let a = true in 
             let b = false in 
             if a && b then 1 else 0",
        );
    }

    #[test]
    fn test_build_permutation_constraint_polynomial_function_composition() {
        test_build_permutation_constraint_polynomial_direct_impl(
            "let compose = fun f -> fun g -> fun x -> f (g x) in
            let add1 = fun x -> x + 1 in
            let mul2 = fun x -> x * 2 in
            compose add1 mul2 5",
        );
    }
    
    #[test]
    fn test_build_permutation_constraint_polynomial_simple() {
        test_build_permutation_constraint_polynomial_direct_impl("1+2");
    }

    #[test]
    fn test_build_permutation_constraint_polynomial_simple_2() {
        test_build_permutation_constraint_polynomial_direct_impl("1+2*3");
    }

    // Tests for gate constraint polynomial
    #[test]
    fn test_build_gate_constraint_polynomial_simple() {
        test_build_gate_constraint_polynomial_impl("1+2");
    }

    #[test]
    fn test_build_gate_constraint_polynomial_arithmetic() {
        test_build_gate_constraint_polynomial_impl("1+2*3");
    }

    #[test]
    fn test_build_gate_constraint_polynomial_let_binding() {
        test_build_gate_constraint_polynomial_impl("let x = 5 in x + x * x");
    }

    #[test]
    fn test_build_gate_constraint_polynomial_boolean_logic() {
        test_build_gate_constraint_polynomial_impl(
            "let a = true in 
             let b = false in 
             if a && b then 1 else 0"
        );
    }

    #[test]
    fn test_build_gate_constraint_polynomial_function_composition() {
        test_build_gate_constraint_polynomial_impl(
            "let compose = fun f -> fun g -> fun x -> f (g x) in
            let add1 = fun x -> x + 1 in
            let mul2 = fun x -> x * 2 in
            compose add1 mul2 5"
        );
    }
}
