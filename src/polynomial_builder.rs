// This module is responsible for building the polynomial constraints for the PLONK circuit.

// Tests for permutation constraint polynomial
// - We need to make sure that the permutation constraint polynomial is all 0 at the points of evaluation
// ((a(X) + βX + γ)(b(X) + βk1X + γ)(c(X) + βk2X + γ)z(X)) - (a(X) + βSσ1 (X) + γ)(b(X) + βSσ2 (X) + γ)(c(X) + βSσ3 (X) + γ)z(Xω))
//

// Tests for the gate constraints polynomial
// Int -> (input_1 - output) * int_selector
// Bool -> (input_1 - output) * bool_selector
// Add -> (input_1 + input_2 - output) * add_selector
// Sub -> (input_1 - input_2 - output) * sub_selector
// Mult -> (input_1 * input_2 - output) * mult_selector
// Div -> (input_1 / input_2 - output) * div_selector
// Eq -> (input_1 == input_2 - output) * eq_selector
// Not -> (input_1 - !input_2) * not_selector
// If -> (input_1 - output) * if_selector
// Let -> (input_1 - output) * let_selector

// TODO: We need to make simple tests not on the evaluated domain to see everything working
use crate::{
    execution_trace::{self, ExecutionTraceTable},
    plonk_circuit::PlonkNodeKind,
    position_cell::{position_to_id, ColumnType, PositionCell},
};
use ark_ff::{FftField, Field};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain, UVPolynomial};
use std::{collections::HashMap, ops::Neg};

#[derive(Debug, Clone)]
    pub struct ExecutionTrace<F: Field, P> {
    pub input1: P,
    pub input2: P,
    pub input3: P,
    pub output: P,
    pub identity_pos1: P, // We need the identity positions for the permutation constraint polynomial
    pub identity_pos2: P,
    pub identity_pos3: P,
    pub identity_out: P,
    pub permutation_input1: P,
    pub permutation_input2: P,
    pub permutation_input3: P,
    pub permutation_output: P,
    pub grand_product: P,
    pub grand_product_random1: F,
    pub grand_product_random2: F,
    pub selectors: HashMap<PlonkNodeKind, P>,
}

pub fn build_execution_trace_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTraceTable<F>,
    grand_product: Vec<F>,
    grand_product_random1: F,
    grand_product_random2: F,
    coset_domain: &D,
) -> ExecutionTrace<F, P> {
    // We need to get the identity positions as vector elements and turn them into a polynomial
    // Compute original wire positions for all 4 columns
    let id1: Vec<F> = (0..grand_product.len())
        .map(|j| {
            F::from(position_to_id(&PositionCell {
                row_idx: j,
                wire_type: ColumnType::Input(0),
            }) as u64)
        })
        .collect();
    let id2: Vec<F> = (0..grand_product.len())
        .map(|j| {
            F::from(position_to_id(&PositionCell {
                row_idx: j,
                wire_type: ColumnType::Input(1),
            }) as u64)
        })
        .collect();
    let id3: Vec<F> = (0..grand_product.len())
        .map(|j| {
            F::from(position_to_id(&PositionCell {
                row_idx: j,
                wire_type: ColumnType::Input(2),
            }) as u64)
        })
        .collect();
    let out_id: Vec<F> = (0..grand_product.len())
        .map(|j| {
            F::from(position_to_id(&PositionCell {
                row_idx: j,
                wire_type: ColumnType::Output,
            }) as u64)
        })
        .collect();

    // Interpolate execution trace columns into polynomials using inverse FFT
    let grand_product_polynomial = P::from_coefficients_vec(
        coset_domain.ifft(&grand_product[0..grand_product.len() - 1]),
    );
    ExecutionTrace {
        input1: P::from_coefficients_vec(coset_domain.ifft(&execution_trace.input1)),
        input2: P::from_coefficients_vec(coset_domain.ifft(&execution_trace.input2)),
        input3: P::from_coefficients_vec(coset_domain.ifft(&execution_trace.input3)),
        output: P::from_coefficients_vec(coset_domain.ifft(&execution_trace.output)),
        identity_pos1: P::from_coefficients_vec(coset_domain.ifft(&id1)),
        identity_pos2: P::from_coefficients_vec(coset_domain.ifft(&id2)),
        identity_pos3: P::from_coefficients_vec(coset_domain.ifft(&id3)),
        identity_out: P::from_coefficients_vec(coset_domain.ifft(&out_id)),
        permutation_input1: P::from_coefficients_vec(
            coset_domain.ifft(&execution_trace.permutation_input1),
        ),
        permutation_input2: P::from_coefficients_vec(
            coset_domain.ifft(&execution_trace.permutation_input2),
        ),
        permutation_input3: P::from_coefficients_vec(
            coset_domain.ifft(&execution_trace.permutation_input3),
        ),
        permutation_output: P::from_coefficients_vec(
            coset_domain.ifft(&execution_trace.permutation_output),
        ),
        grand_product: P::from_coefficients_vec(
            coset_domain.ifft(&grand_product[0..grand_product.len() - 1]),
        ),
        grand_product_random1: grand_product_random1,
        grand_product_random2: grand_product_random2,
        selectors: execution_trace
            .selectors
            .iter()
            .map(|(k, v)| (k.clone(), P::from_coefficients_vec(coset_domain.ifft(v))))
            .collect(),
    }
}

pub fn build_execution_trace_extended_evaluation_form<F: FftField, P: UVPolynomial<F>>(
    execution_trace: &ExecutionTrace<F, P>,
    coset_domain: &Radix2EvaluationDomain<F>,
    evaluation_domain: &Radix2EvaluationDomain<F>,
) -> ExecutionTrace<F, Vec<F>> {
    // Evaluate polynomials over the full domain
    let domain_elements: Vec<F> = evaluation_domain.elements().collect();

    // TODO: Optimize using FFT
    ExecutionTrace {
        input1: domain_elements
            .iter()
            .map(|x| execution_trace.input1.evaluate(x))
            .collect(),
        input2: domain_elements
            .iter()
            .map(|x| execution_trace.input2.evaluate(x))
            .collect(),
        input3: domain_elements
            .iter()
            .map(|x| execution_trace.input3.evaluate(x))
            .collect(),
        output: domain_elements
            .iter()
            .map(|x| execution_trace.output.evaluate(x))
            .collect(),
        identity_pos1: domain_elements
            .iter()
            .map(|x| execution_trace.identity_pos1.evaluate(x))
            .collect(),
        identity_pos2: domain_elements
            .iter()
            .map(|x| execution_trace.identity_pos2.evaluate(x))
            .collect(),
        identity_pos3: domain_elements
            .iter()
            .map(|x| execution_trace.identity_pos3.evaluate(x))
            .collect(),
        identity_out: domain_elements
            .iter()
            .map(|x| execution_trace.identity_out.evaluate(x))
            .collect(),
        permutation_input1: domain_elements
            .iter()
            .map(|x| execution_trace.permutation_input1.evaluate(x))
            .collect(),
        permutation_input2: domain_elements
            .iter()
            .map(|x| execution_trace.permutation_input2.evaluate(x))
            .collect(),
        permutation_input3: domain_elements
            .iter()
            .map(|x| execution_trace.permutation_input3.evaluate(x))
            .collect(),
        permutation_output: domain_elements
            .iter()
            .map(|x| execution_trace.permutation_output.evaluate(x))
            .collect(),
        grand_product: domain_elements
            .iter()
            .map(|x| execution_trace.grand_product.evaluate(x))
            .collect(),
        grand_product_random1: execution_trace.grand_product_random1,
        grand_product_random2: execution_trace.grand_product_random2,
        selectors: execution_trace
            .selectors
            .iter()
            .map(|(k, p)| {
                (
                    k.clone(),
                    domain_elements.iter().map(|x| p.evaluate(x)).collect(),
                )
            })
            .collect(),
    }
}

/// Builds the addition constraint polynomial: (input_1 + input_2 - output) * add_selector
fn build_addition_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,  // Unused for gate constraints
    _gamma: F, // Unused for gate constraints
) -> P {
    let add_selector = execution_trace.selectors.get(&PlonkNodeKind::Add).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.input2)
        .zip(&execution_trace.output)
        .zip(add_selector)
        .map(|(((a, b), c), s)| (*a + *b - *c) * *s)
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the subtraction constraint polynomial: (input_1 - input_2 - output) * sub_selector
fn build_subtraction_constraint_polynomial<
    F: FftField,
    P: UVPolynomial<F>,
    D: EvaluationDomain<F>,
>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let sub_selector = execution_trace.selectors.get(&PlonkNodeKind::Sub).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.input2)
        .zip(&execution_trace.output)
        .zip(sub_selector)
        .map(|(((a, b), c), s)| (*a - *b - *c) * *s)
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the multiplication constraint polynomial: (input_1 * input_2 - output) * mult_selector
fn build_multiplication_constraint_polynomial<
    F: FftField,
    P: UVPolynomial<F>,
    D: EvaluationDomain<F>,
>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let mult_selector = execution_trace.selectors.get(&PlonkNodeKind::Mult).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.input2)
        .zip(&execution_trace.output)
        .zip(mult_selector)
        .map(|(((a, b), c), s)| (*a * *b - *c) * *s)
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the division constraint polynomial: (input_1 - input_2 * output) * div_selector
fn build_division_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let div_selector = execution_trace.selectors.get(&PlonkNodeKind::Div).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.input2)
        .zip(&execution_trace.output)
        .zip(div_selector)
        .map(|(((a, b), c), s)| (*a - *b * *c) * *s)
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the equality constraint polynomial: (input_1 - input_2) * (1 - output) * eq_selector
fn build_equality_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let eq_selector = execution_trace.selectors.get(&PlonkNodeKind::Eq).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.input2)
        .zip(&execution_trace.output)
        .zip(eq_selector)
        .map(|(((a, b), c), s)| {
            let diff = *a - *b;
            let one = F::one();
            (diff * (one - *c)) * *s
        })
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the negation constraint polynomial: (input_1 - !output) * not_selector
/// where !output is implemented as 1 - output for field elements
fn build_negation_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let not_selector = execution_trace.selectors.get(&PlonkNodeKind::Not).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.output)
        .zip(not_selector)
        .map(|((a, c), s)| {
            let one = F::one();
            (*a - (one - *c)) * *s
        })
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the if constraint polynomial: (input_1 * input_2 + (1-input_1) * input_3 - output) * if_selector
fn build_if_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let if_selector = execution_trace.selectors.get(&PlonkNodeKind::If).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.input2)
        .zip(&execution_trace.input3)
        .zip(&execution_trace.output)
        .zip(if_selector)
        .map(|((((cond, then_val), else_val), out), s)| {
            let one = F::one();
            let then_term = *cond * *then_val;
            let else_term = (one - *cond) * *else_val;
            (then_term + else_term - *out) * *s
        })
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the integer constraint polynomial: (input_1 - output) * int_selector
fn build_int_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let int_selector = execution_trace.selectors.get(&PlonkNodeKind::Int).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.output)
        .zip(int_selector)
        .map(|((a, c), s)| (*a - *c) * *s)
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the boolean constraint polynomial: (input_1 - output) * bool_selector
fn build_bool_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    _beta: F,
    _gamma: F,
) -> P {
    let bool_selector = execution_trace.selectors.get(&PlonkNodeKind::Bool).unwrap();
    let constraint_evals: Vec<F> = execution_trace
        .input1
        .iter()
        .zip(&execution_trace.output)
        .zip(bool_selector)
        .map(|((a, c), s)| (*a - *c) * *s)
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the complete gate constraint polynomial by combining all individual gate constraints
pub fn build_gate_constraint_polynomial<F: FftField, P: UVPolynomial<F>, D: EvaluationDomain<F>>(
    execution_trace: &ExecutionTrace<F, Vec<F>>,
    domain: &D,
    beta: F,
    gamma: F,
) -> P {
    // Build individual constraint polynomials
    let int_poly = build_int_constraint_polynomial(execution_trace, domain, beta, gamma);
    let bool_poly = build_bool_constraint_polynomial(execution_trace, domain, beta, gamma);
    let add_poly = build_addition_constraint_polynomial(execution_trace, domain, beta, gamma);
    let sub_poly = build_subtraction_constraint_polynomial(execution_trace, domain, beta, gamma);
    let mult_poly =
        build_multiplication_constraint_polynomial(execution_trace, domain, beta, gamma);
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

/// Builds the permutation constraint polynomial for PLONK's permutation argument.
///
///
/// The grand product polynomial Z has length n+1 (where n is the number of rows),
/// with Z[0] = 1 and for j in [0..n-1]:
///     Z[j+1] = Z[j]
///             * Π(i=0..m-1) [ vᵢ(j) + β * vᵢ(σᵢ(j)) + γ ]
///             / Π(i=0..m-1) [ vᵢ(j) + β * δᵢ(j) + γ ]
///
/// Here:
///   • vᵢ(j) is the i-th wire value at row j,
///   • vᵢ(σᵢ(j)) is that i-th wire evaluated at the permuted index from permutation_table,
///   • δᵢ(j) is the wire's domain factor (position ID in our implementation).
///
/// Note: All polynomials are in evaluation form over domain D
pub fn build_permutation_constraint_polynomial<
    F: FftField,
    P: UVPolynomial<F>,
    D: EvaluationDomain<F>,
>(
    input1_wire: &Vec<F>,
    input2_wire: &Vec<F>,
    input3_wire: &Vec<F>,
    output_wire: &Vec<F>,
    s_sigma1: &Vec<F>,
    s_sigma2: &Vec<F>,
    s_sigma3: &Vec<F>,
    sigma_out: &Vec<F>,
    z: &Vec<F>,
    z_shifted: &Vec<F>,
    id1: &Vec<F>,
    id2: &Vec<F>,
    id3: &Vec<F>,
    out_id: &Vec<F>,
    domain: &D,
    random1: F,
    random2: F,
) -> P {
    let domain_elements: Vec<F> = domain.elements().collect();
    let random2_vec = vec![random2; domain_elements.len()];

    // Compute left side terms (original wire positions)
    // Left side: (a(X) + β*id1(X) + γ)(b(X) + β*id2(X) + γ)(c(X) + β*id3(X) + γ)(d(X) + β*id4(X) + γ)z(X)
    let non_permutated_terms = [
        add_vectors(
            &add_vectors(input1_wire, &scalar_multiply(&random1, id1)),
            &random2_vec,
        ),
        add_vectors(
            &add_vectors(input2_wire, &scalar_multiply(&random1, id2)),
            &random2_vec,
        ),
        add_vectors(
            &add_vectors(input3_wire, &scalar_multiply(&random1, id3)),
            &random2_vec,
        ),
        add_vectors(
            &add_vectors(output_wire, &scalar_multiply(&random1, out_id)),
            &random2_vec,
        ),
    ];

    // Compute right side terms (permuted positions)
    // Right side: (a(X) + βSσ₁(X) + γ)(b(X) + βSσ₂(X) + γ)(c(X) + βSσ₃(X) + γ)(d(X) + βSσ₄(X) + γ)z(Xω)
    let permuted_terms = [
        add_vectors(
            &add_vectors(input1_wire, &scalar_multiply(&random1, s_sigma1)),
            &random2_vec,
        ),
        add_vectors(
            &add_vectors(input2_wire, &scalar_multiply(&random1, s_sigma2)),
            &random2_vec,
        ),
        add_vectors(
            &add_vectors(input3_wire, &scalar_multiply(&random1, s_sigma3)),
            &random2_vec,
        ),
        add_vectors(
            &add_vectors(output_wire, &scalar_multiply(&random1, sigma_out)),
            &random2_vec,
        ),
    ];

    // Calculate left and right products including all 4 terms
    let left_product = non_permutated_terms
        .iter()
        .fold(z_shifted.clone(), |acc, term| multiply_vectors(&acc, term));

    let right_product = permuted_terms
        .iter()
        .fold(z.clone(), |acc, term| multiply_vectors(&acc, term));

    // Final constraint: left_product - right_product
    // This is the permutation constraint polynomial:
    // (a(X) + β*id1(X) + γ)(b(X) + β*id2(X) + γ)(c(X) + β*id3(X) + γ)(d(X) + β*id4(X) + γ)z(X)
    // - (a(X) + βSσ₁(X) + γ)(b(X) + βSσ₂(X) + γ)(c(X) + βSσ₃(X) + γ)(d(X) + βSσ₄(X) + γ)z(Xω)
    let constraint_evals: Vec<F> = left_product
        .iter()
        .zip(&right_product)
        .map(|(l, r)| *l - *r)
        .collect();

    P::from_coefficients_vec(domain.ifft(&constraint_evals))
}

/// Builds the permutation constraint polynomial for PLONK's permutation argument.
///
///
/// The grand product polynomial Z has length n+1 (where n is the number of rows),
/// with Z[0] = 1 and for j in [0..n-1]:
///     Z[j+1] = Z[j]
///             * Π(i=0..m-1) [ vᵢ(j) + β * vᵢ(σᵢ(j)) + γ ]
///             / Π(i=0..m-1) [ vᵢ(j) + β * δᵢ(j) + γ ]
///
/// Here:
///   • vᵢ(j) is the i-th wire value at row j,
///   • vᵢ(σᵢ(j)) is that i-th wire evaluated at the permuted index from permutation_table,
///   • δᵢ(j) is the wire's domain factor (position ID in our implementation).
///
/// Note: This function works directly with polynomials rather than evaluation vectors
pub fn build_permutation_constraint_polynomial_no_fft<
    F: FftField
>(
    execution_trace: &ExecutionTrace<F, DensePolynomial<F>>,
    domain: &Radix2EvaluationDomain<F>,
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
    let term1_left = (input1_poly + &(id1_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Term 2: b(X) + β*id2(X) + γ
    let term2_left = (input2_poly + &(id2_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Term 3: c(X) + β*id3(X) + γ
    let term3_left = (input3_poly + &(id3_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Term 4: d(X) + β*id4(X) + γ
    let term4_left = (output_poly + &(out_id_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Term 1: a(X) + βSσ₁(X) + γ
    let term1_right = (input1_poly + &(sigma1_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Term 2: b(X) + βSσ₂(X) + γ
    let term2_right = (input2_poly + &(sigma2_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Term 3: c(X) + βSσ₃(X) + γ
    let term3_right = (input3_poly + &(sigma3_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Term 4: d(X) + βSσ₄(X) + γ
    let term4_right = (output_poly + &(sigma_out_poly * random1)) + DensePolynomial::from_coefficients_vec(vec![random2]);
    
    // Use z(X) on left and z(ωX) on right to match standard PLONK formula
    
    // Left side: product of non-permuted terms with z(X)
    let left_term = &(&(&(&term1_left * &term2_left) * &term3_left) * &term4_left);
    
    // Right side: product of permuted terms with z(ωX) 
    let right_term = &(&(&(&term1_right * &term2_right) * &term3_right) * &term4_right);
    
    // Final constraint: z(X)⋅left_term - z(ωX)⋅right_term = 0
    let constraint = &(z_poly * left_term) - &(&z_shifted_poly * right_term);
    
    constraint
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution_trace::parse_to_execution_trace,
        grand_product::build_grand_product_evaluation_vector,
    };
    use ark_bn254::Fr as F;
    use ark_ff::{UniformRand, Zero};
    use ark_poly::{univariate::DensePolynomial, Polynomial};
    use ark_std::test_rng;

    fn test_build_permutation_constraint_polynomial(input: &str) {
        let execution_trace_table = parse_to_execution_trace(input);

        // Generate random challenges
        let random1 = F::rand(&mut test_rng());
        let random2 = F::rand(&mut test_rng());

        // Create coset domain (base size)
        let coset_domain_size = execution_trace_table.input1.len().next_power_of_two();
        let coset_domain = ark_poly::Radix2EvaluationDomain::<F>::new(coset_domain_size)
            .expect("Failed to create coset domain");

        for i in 0..coset_domain.size() - 1 {
            let expected = coset_domain.element(i) * coset_domain.group_gen;
            let actual = coset_domain.element(i + 1);
            assert_eq!(
                actual, expected,
                "Coset domain property violated: coset_domain.element(i+1) != coset_domain.element(i) * coset_domain.element(0) at i={}",
                i
            );
        }
        // Create evaluation domain (4x coset size)
        // let evaluation_domain_size = coset_domain_size * 4;
        // let evaluation_domain = ark_poly::Radix2EvaluationDomain::<F>::new(evaluation_domain_size)
        //     .expect("Failed to create evaluation domain");

        // Build grand product vector
        let grand_product_vector = build_grand_product_evaluation_vector(
            &execution_trace_table,
            Some(random1),
            Some(random2),
        );

        // Build polynomial form using coset domain
        let trace_poly: ExecutionTrace<F, DensePolynomial<F>> = build_execution_trace_polynomial(
            &execution_trace_table,
            grand_product_vector,
            random1,
            random2,
            &coset_domain,
        );



        // Convert to evaluation form over evaluation domain
        // let extended_trace_eval =
        //     build_execution_trace_extended_evaluation_form(&trace_poly, &coset_domain, &evaluation_domain);

        // Compute the shifted grand product polynomial
        // We need to shift the grand product polynomial by multiplying each input by the generator
        // of the coset domain
        // let coset_gen = coset_domain.group_gen;
        // let mut grand_product_shifted = Vec::with_capacity(evaluation_domain.size());
        
        // // For each element in the evaluation domain
        // for i in 0..evaluation_domain.size() {
        //     // Evaluate the grand product polynomial at x * coset_gen
        //     let x = evaluation_domain.element(i);
        //     let shifted_point = x * coset_gen;
        //     let shifted_eval = trace_poly.grand_product.evaluate(&shifted_point);
        //     grand_product_shifted.push(shifted_eval);
        // }
        
        // Build constraint polynomial using evaluation domain
        // let constraint_poly : DensePolynomial<F> = build_permutation_constraint_polynomial(
        //     &extended_trace_eval.input1,
        //     &extended_trace_eval.input2,
        //     &extended_trace_eval.input3,
        //     &extended_trace_eval.output,
        //     &extended_trace_eval.permutation_input1,
        //     &extended_trace_eval.permutation_input2,
        //     &extended_trace_eval.permutation_input3,
        //     &extended_trace_eval.permutation_output,
        //     &extended_trace_eval.grand_product,
        //     &grand_product_shifted,
        //     &extended_trace_eval.identity_pos1,
        //     &extended_trace_eval.identity_pos2,
        //     &extended_trace_eval.identity_pos3,
        //     &extended_trace_eval.output,
        //     &evaluation_domain,
        //     random1,
        //     random2,
        // );

        let constraint_poly = build_permutation_constraint_polynomial_no_fft(&trace_poly, &coset_domain, random1, random2);

        // Evaluate at all points in the evaluation domain
        let constraint_evals = coset_domain
            .elements()
            .map(|x| constraint_poly.evaluate(&x))
            .collect::<Vec<_>>();
        println!("constraint_evals: {:?}", constraint_evals);
        // Validation checks
        for eval in constraint_evals {
            assert!(eval.is_zero(), "Constraint polynomial should evaluate to 0");
        }
    }

    #[test]
    fn test_build_permutation_constraint_polynomial_let_x_5_in_x_plus_x_times_x() {
        test_build_permutation_constraint_polynomial("let x = 5 in x + x * x");
    }

    #[test]
    fn test_build_permutation_constraint_polynomial_boolean_logic() {
        test_build_permutation_constraint_polynomial(
            "let a = true in 
             let b = false in 
             if a && b then 1 else 0",
        );
    }

    #[test]
    fn test_build_permutation_constraint_polynomial_function_composition() {
        test_build_permutation_constraint_polynomial(
            "let compose = fun f -> fun g -> fun x -> f (g x) in
            let add1 = fun x -> x + 1 in
            let mul2 = fun x -> x * 2 in
            compose add1 mul2 5",
        );
    }
    #[test]
    fn test_build_permutation_constraint_polynomial_simple() {
        test_build_permutation_constraint_polynomial("1+2");
    }

    #[test]
    fn test_build_permutation_constraint_polynomial_simple_2() {
        test_build_permutation_constraint_polynomial("1+2*3");
    }

    // fn test_build_gate_constraint_polynomial(input: &str) {
    //     let execution_trace_table = parse_to_execution_trace(input);

    //     // Generate random challenges (though not strictly needed for gate constraints)
    //     let beta = F::rand(&mut test_rng());
    //     let gamma = F::rand(&mut test_rng());

    //     // Create coset domain (base size)
    //     let coset_domain_size = execution_trace_table.input1.len().next_power_of_two();
    //     let coset_domain = ark_poly::GeneralEvaluationDomain::<F>::new(coset_domain_size)
    //         .expect("Failed to create coset domain");

    //     // Create evaluation domain (4x coset size for better interpolation)
    //     let evaluation_domain_size = coset_domain_size * 4;
    //     let evaluation_domain = ark_poly::GeneralEvaluationDomain::<F>::new(evaluation_domain_size)
    //         .expect("Failed to create evaluation domain");

    //     // Build grand product vector (needed for full trace construction)
    //     let grand_product_vector = build_grand_product_evaluation_vector(
    //         &execution_trace_table,
    //         Some(beta),
    //         Some(gamma)
    //     );

    //     // Build polynomial form using coset domain
    //     let trace_poly: ExecutionTrace<DensePolynomial<F>> = build_execution_trace_polynomial(
    //         &execution_trace_table,
    //         grand_product_vector,
    //         &coset_domain
    //     );

    //     // Convert to evaluation form over evaluation domain
    //     let trace_eval = build_execution_trace_extended_evaluation_form(
    //         &trace_poly,
    //         &evaluation_domain
    //     );

    //     // Build gate constraint polynomial
    //     let constraint_poly = build_gate_constraint_polynomial::<F, DensePolynomial<F>, _>(
    //         &trace_eval,
    //         &evaluation_domain,
    //         beta,
    //         gamma
    //     );

    //     // Evaluate at all points in the evaluation domain
    //     let constraint_evals = evaluation_domain.fft(&constraint_poly.coeffs);

    //     // Validation checks - all evaluations should be zero
    //     for eval in constraint_evals {
    //         assert!(eval.is_zero(), "Gate constraint polynomial should evaluate to 0");
    //     }
    // }

    // #[test]
    // fn test_arithmetic_operations() {
    //     test_build_gate_constraint_polynomial(
    //         "let x = 5 in
    //          let y = 3 in
    //          x + y * (x - y)"
    //     );
    // }

    // #[test]
    // fn test_boolean_operations() {
    //     test_build_gate_constraint_polynomial(
    //         "let a = true in
    //          let b = false in
    //          if a && !b then 1 else 0"
    //     );
    // }

    // #[test]
    // fn test_equality_comparison() {
    //     test_build_gate_constraint_polynomial(
    //         "let x = 5 in
    //          let y = 5 in
    //          x == y"
    //     );
    // }

    // #[test]
    // fn test_complex_expression() {
    //     test_build_gate_constraint_polynomial(
    //         "let x = 10 in
    //          let y = 2 in
    //          let z = x / y in
    //          if z == 5 then x + y else x - y"
    //     );
    // }
}

/// Multiplies two vectors pointwise
fn multiply_vectors<F: Field>(v1: &[F], v2: &[F]) -> Vec<F> {
    v1.iter().zip(v2.iter()).map(|(a, b)| *a * *b).collect()
}

/// Adds two vectors pointwise
fn add_vectors<F: Field>(v1: &[F], v2: &[F]) -> Vec<F> {
    v1.iter().zip(v2.iter()).map(|(a, b)| *a + *b).collect()
}

/// Multiplies a vector by a scalar
fn scalar_multiply<F: Field>(scalar: &F, v: &[F]) -> Vec<F> {
    v.iter().map(|a| *scalar * *a).collect()
}
