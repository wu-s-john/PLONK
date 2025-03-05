use crate::{
    execution_trace::ExecutionTraceTable,
    plonk_circuit::PlonkNodeKind,
    position_cell::{position_to_id, ColumnType, PositionCell},
};
use ark_ff::{FftField, Field};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain
};
use std::collections::HashMap;

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

pub fn build_execution_trace_polynomial<F: FftField, P: DenseUVPolynomial<F>, D: EvaluationDomain<F>>(
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
    let grand_product_polynomial =
        P::from_coefficients_vec(coset_domain.ifft(&grand_product[0..grand_product.len() - 1]));
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
        grand_product: grand_product_polynomial,
        grand_product_random1,
        grand_product_random2,
        selectors: execution_trace
            .selectors
            .iter()
            .map(|(k, v)| (k.clone(), P::from_coefficients_vec(coset_domain.ifft(v))))
            .collect(),
    }
}

pub fn build_execution_trace_extended_evaluation_form<F: FftField, P: DenseUVPolynomial<F>>(
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