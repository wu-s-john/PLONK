//! This file is responsible for building the grand product polynomial representing a permutation
//! and building the constraint polynomial for it. The grand product polynomial encodes the
//! permutation cycles between wire positions, while the constraint polynomial ensures the
//! permutation relationship is satisfied.

use crate::{execution_trace::{AllOpTraces, ExecutionTrace, GateEvaluationTable, WireMap, WirePosition, WireType}, offset_table::OffsetTable, polynomial_utils::evaluations_to_dense_polynomial};
use ark_ff::{FftField, Field};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
use std::collections::HashMap;

pub type PermutationTrace = AllOpTraces<WirePosition>;
type FlatPermutationMap = WireMap<WirePosition>;

/// Build a permutation trace (sigma) by linking each group of wire positions
/// that share the same node_id in a cycle. For example, if node_id=17 appears in
/// four distinct wire positions [p0, p1, p2, p3], we arrange them in a cycle:
/// p0→p1, p1→p2, p2→p3, p3→p0.
pub fn build_permutation_trace<F>(trace: &ExecutionTrace<F>) -> FlatPermutationMap {
    let grouped = trace.group_wire_positions_by_node_id();
    let mut permutation = HashMap::new();

    for (_node_id, positions) in grouped {
        let k = positions.len();
        if k == 1 {
            // Trivial cycle: only one wire position
            let p0 = positions[0].clone();
            permutation.insert(p0.clone(), p0);
        } else {
            // Link them in a cycle
            for i in 0..k {
                let current = positions[i].clone();
                let next = positions[(i + 1) % k].clone();
                permutation.insert(current, next);
            }
        }
    }

    permutation
}

pub fn build_gate_map<F: Field>(trace: &ExecutionTrace<F>) -> WireMap<usize> {
    let mut value_map = HashMap::new();
    for (wire_pos, node_id) in trace.collect_wire_positions() {
        value_map.insert(wire_pos, node_id);
    }
    value_map
}

/// A struct representing how each permutation wire is mapped to the next wire in the permutation cycle.
/// Each vector represents σ₁, σ₂, σ₃, σ₄ in Plonk terminology.
#[derive(Debug, Clone)]
pub struct PermutationTable<F> {
    /// σ₁: permutation for first input wire
    pub sigma1: Vec<F>,
    /// σ₂: permutation for second input wire
    pub sigma2: Vec<F>,
    /// σ₃: permutation for third input wire
    pub sigma3: Vec<F>,
    /// σ₄: permutation for output wire
    pub sigma_output: Vec<F>,
}

impl<F> PermutationTable<F> {
    pub fn new(size: usize, default: F) -> Self
    where
        F: Clone,
    {
        Self {
            sigma1: vec![default.clone(); size],
            sigma2: vec![default.clone(); size],
            sigma3: vec![default.clone(); size],
            sigma_output: vec![default.clone(); size],
        }
    }
}

/// Build permutation polynomials from a flat permutation map.
/// For each position in the circuit, compute its "next" position in the permutation cycle.
pub fn build_permutation_table<F>(
    trace: &ExecutionTrace<F>,
    offset_table: &OffsetTable,
    perm_map: &FlatPermutationMap,
) -> PermutationTable<usize>
where
    F: Clone
{
    let mut poly = PermutationTable::new(offset_table.total_rows, 0);

    // Helper function to get the next index in the permutation
    let get_next_index = |wire: &WirePosition| -> usize {
        let curr_idx = offset_table.to_index(wire);
        perm_map
            .get(wire)
            .map(|next| offset_table.to_index(next))
            .unwrap_or(curr_idx)
    };

    // Process single-input operations
    for (op_kind, table) in &trace.single_input {
        for (row_idx, _row) in table.rows.iter().enumerate() {
            // Current wire positions
            let wire_in = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Input(0),
            };
            let wire_out = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Output,
            };

            // Get next positions in permutation
            let curr_in_idx = offset_table.to_index(&wire_in);
            poly.sigma1[curr_in_idx] = get_next_index(&wire_in);

            let curr_out_idx = offset_table.to_index(&wire_out);
            poly.sigma_output[curr_out_idx] = get_next_index(&wire_out);

            // For unused wires in single-input ops, point to self
            poly.sigma2[curr_in_idx] = curr_in_idx;
            poly.sigma3[curr_in_idx] = curr_in_idx;
        }
    }

    // Process double-input operations
    for (op_kind, table) in &trace.double_input {
        for (row_idx, _row) in table.rows.iter().enumerate() {
            // Current wire positions
            let wire_in0 = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Input(0),
            };
            let wire_in1 = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Input(1),
            };
            let wire_out = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Output,
            };

            // Get next positions in permutation
            let curr_in0_idx = offset_table.to_index(&wire_in0);
            poly.sigma1[curr_in0_idx] = get_next_index(&wire_in0);

            let curr_in1_idx = offset_table.to_index(&wire_in1);
            poly.sigma2[curr_in1_idx] = get_next_index(&wire_in1);

            let curr_out_idx = offset_table.to_index(&wire_out);
            poly.sigma_output[curr_out_idx] = get_next_index(&wire_out);

            // For unused wire in double-input ops, point to self
            poly.sigma3[curr_in0_idx] = curr_in0_idx; // This maybe wrong
        }
    }

    // Process triple-input operations
    for (op_kind, table) in &trace.triple_input {
        for (row_idx, _row) in table.rows.iter().enumerate() {
            // Current wire positions
            let wire_in0 = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Input(0),
            };
            let wire_in1 = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Input(1),
            };
            let wire_in2 = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Input(2),
            };
            let wire_out = WirePosition {
                op_kind: op_kind.clone(),
                row_idx,
                wire_type: WireType::Output,
            };

            // Get next positions in permutation
            let curr_in0_idx = offset_table.to_index(&wire_in0);
            poly.sigma1[curr_in0_idx] = get_next_index(&wire_in0);

            let curr_in1_idx = offset_table.to_index(&wire_in1);
            poly.sigma2[curr_in1_idx] = get_next_index(&wire_in1);

            let curr_in2_idx = offset_table.to_index(&wire_in2);
            poly.sigma3[curr_in2_idx] = get_next_index(&wire_in2);

            let curr_out_idx = offset_table.to_index(&wire_out);
            poly.sigma_output[curr_out_idx] = get_next_index(&wire_out);
        }
    }

    poly
}

/// Builds the grand product polynomial Z.  
/// Z has length n+1 (where n is the number of rows in the evaluation table),  
/// with Z[0] = 1 and then for j in [0..n-1]:
///     Z[j+1] = Z[j]
///             * Π(i=0..m-1) [ vᵢ(j) + β * vᵢ(σᵢ(j)) + γ ]
///             / Π(i=0..m-1) [ vᵢ(j) + β * δᵢ(j) + γ ]
///
/// Here:
///   • vᵢ(j) is the i-th wire value at row j,
///   • vᵢ(σᵢ(j)) is that i-th wire evaluated at the permuted index from permutation_table,
///   • δᵢ(j) is a placeholder for the shift (or domain factor) in a real Plonk system.
///     For simplicity, we just use j (the row index) or some other scheme below.
///
pub fn build_grand_product_evaluation_polynomial<F: Field>(
    permutation_table: &PermutationTable<usize>,
    evaluation_table: &GateEvaluationTable<F>,
    beta: F,
    gamma: F,
) -> Vec<F> {
    let n = evaluation_table.input1.len();
    let mut z = vec![F::one(); n + 1];

    // For each row j, compute the ratio of products
    for j in 0..n {
        // Collect wire values at row j
        let v1_j = evaluation_table.input1[j];
        let v2_j = evaluation_table.input2[j];
        let v3_j = evaluation_table.input3[j];
        let vo_j = evaluation_table.output[j];

        // Collect wire values at row sigmaᵢ(j)
        let v1_sigma = evaluation_table.input1[permutation_table.sigma1[j]];
        let v2_sigma = evaluation_table.input2[permutation_table.sigma2[j]];
        let v3_sigma = evaluation_table.input3[permutation_table.sigma3[j]];
        let vo_sigma = evaluation_table.output[permutation_table.sigma_output[j]];

        // Numerator = Π over each wire i of [ vᵢ(j) + β * vᵢ(σᵢ(j)) + γ ]
        let mut numerator = F::one();
        numerator *= v1_j + (beta * v1_sigma) + gamma;
        numerator *= v2_j + (beta * v2_sigma) + gamma;
        numerator *= v3_j + (beta * v3_sigma) + gamma;
        numerator *= vo_j + (beta * vo_sigma) + gamma;

        // Denominator = Π over each wire i of [ vᵢ(j) + β * δᵢ(j) + γ ]
        // For simplicity, we use j as part of δᵢ(j) (you could refine this if needed).
        let j_as_field = F::from(j as u64);
        let mut denominator = F::one();
        denominator *= v1_j + (beta * j_as_field) + gamma;
        denominator *= v2_j + (beta * j_as_field) + gamma;
        denominator *= v3_j + (beta * j_as_field) + gamma;
        denominator *= vo_j + (beta * j_as_field) + gamma;

        // Update Z[j+1] = Z[j] * (numerator / denominator)
        // (We assume denominator != 0 in typical Plonk usage.)
        z[j + 1] = z[j] * numerator * denominator.inverse().unwrap();
    }
    z
}

pub struct PermutationPolynomials<F: Field> {
    pub sigma1: DensePolynomial<F>,
    pub sigma2: DensePolynomial<F>,
    pub sigma3: DensePolynomial<F>,
    pub sigma_output: DensePolynomial<F>,
    pub z: DensePolynomial<F>,
}


pub fn build_permutation_polynomials<F: FftField>(
    permutation_table: &PermutationTable<usize>,
    evaluation_table: &GateEvaluationTable<F>,
    beta: F,
    gamma: F,
    coset_domain: &GeneralEvaluationDomain<F>,
) -> PermutationPolynomials<F> {
    let convert_to_poly = |indices: &[usize]| -> DensePolynomial<F> {
        evaluations_to_dense_polynomial(
            &indices.iter().map(|&x| F::from(x as u64)).collect::<Vec<_>>(),
            coset_domain
        )
    };

    let sigma1 = convert_to_poly(&permutation_table.sigma1);
    let sigma2 = convert_to_poly(&permutation_table.sigma2); 
    let sigma3 = convert_to_poly(&permutation_table.sigma3);
    let sigma_output = convert_to_poly(&permutation_table.sigma_output);
    let z = build_grand_product_evaluation_polynomial(permutation_table, evaluation_table, beta, gamma);
    PermutationPolynomials {
        sigma1,
        sigma2,
        sigma3,
        sigma_output,
        z: evaluations_to_dense_polynomial(&z, coset_domain),
    }
}

/// Builds the permutation constraint polynomial E(X), whose i-th evaluation is:
/// E(ω^i) = Z(ω^(i+1)) * ∏(wᵢ(j) + β * j + γ)  −  Z(ω^i) * ∏(wᵢ(j) + β * σᵢ(j) + γ)
///
/// Here,
///   • ω is a primitive n-th root of unity (for the evaluation_domain),
///   • Z(X) is the grand product polynomial,
///   • wᵢ(j) are the wire polynomials evaluated at row j (or at ω^j),
///   • σᵢ(j) are the permutation polynomials evaluated at ω^j,
///   • β, γ are Plonk's permutation coefficients.
/// 
/// The result is returned in coefficient form as a DensePolynomial<F>.
pub fn build_permutation_constraint_evaluation_polynomial<F: FftField>(
    polynomials: &PermutationPolynomials<F>,
    evaluation_table: &GateEvaluationTable<F>,
    beta: F,
    gamma: F,
    // A domain with "unit steps": {1, ω, ω^2, ... , ω^(n-1)}
    evaluation_domain: &GeneralEvaluationDomain<F>,
) -> Vec<F> {
    let n = evaluation_domain.size();
    // 1) Evaluate Z and σ polynomials at each point ω^i in the evaluation_domain.
    let z_evals = polynomials.z.evaluate_over_domain_by_ref(*evaluation_domain);
    let sigma1_evals = polynomials.sigma1.evaluate_over_domain_by_ref(*evaluation_domain);
    let sigma2_evals = polynomials.sigma2.evaluate_over_domain_by_ref(*evaluation_domain);
    let sigma3_evals = polynomials.sigma3.evaluate_over_domain_by_ref(*evaluation_domain);
    let sigma_output_evals = polynomials.sigma_output.evaluate_over_domain_by_ref(*evaluation_domain);

    // 2) For each i, compute wire values wᵢ(ω^i). In this basic implementation, we'll
    //    assume evaluation_table's values align 1–1 with these domain points.
    //    (i.e. w1(ω^i) = evaluation_table.input1[i], etc.)
    //    If your actual code uses interpolation or a coset shift, adjust as needed.
    let input1 = &evaluation_table.input1;
    let input2 = &evaluation_table.input2;
    let input3 = &evaluation_table.input3;
    let output = &evaluation_table.output;

    // 3) Build E(ω^i) for i in [0..n):
    //      E(ω^i) = Z(ω^(i+1)) * ∏(wᵢ(i) + β * i + γ) − Z(ω^i) * ∏( wᵢ(i) + β * σᵢ(ω^i) + γ )
    //    Store these E-values in constraint_evals.
    let mut constraint_evals = vec![F::zero(); n];
    for i in 0..n {
        let w1 = input1[i];
        let w2 = input2[i];
        let w3 = input3[i];
        let w4 = output[i];

        let z_next = z_evals[(i + 1) % n];  // Z(ω^(i+1))
        let z_cur = z_evals[i];            // Z(ω^i)

        // Convert i to field for the j index in the "wᵢ + β * j + γ" term
        let i_as_field = F::from(i as u64);

        // ∏(wᵢ(i) + β * i + γ)
        let lhs_product = (w1 + beta * i_as_field + gamma)
            * (w2 + beta * i_as_field + gamma)
            * (w3 + beta * i_as_field + gamma)
            * (w4 + beta * i_as_field + gamma);

        // ∏(wᵢ(i) + β * σᵢ(ω^i) + γ), using the σ-evals
        let s1 = sigma1_evals[i];
        let s2 = sigma2_evals[i];
        let s3 = sigma3_evals[i];
        let s4 = sigma_output_evals[i];

        let rhs_product = (w1 + beta * s1 + gamma)
            * (w2 + beta * s2 + gamma)
            * (w3 + beta * s3 + gamma)
            * (w4 + beta * s4 + gamma);

        // E(ω^i)
        constraint_evals[i] = z_next * lhs_product - z_cur * rhs_product;
    }

    constraint_evals
}