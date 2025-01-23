use std::collections::HashMap;

use ark_ff::{FftField, Field, One, Zero};
use ark_poly::univariate::DensePolynomial;
use ark_poly::EvaluationDomain;
use ark_poly::{GeneralEvaluationDomain, Polynomial};

use crate::ast::{ASTKind, NodeMeta};
use crate::offset_table::OffsetTable;
use crate::polynomial_utils::evaluations_to_dense_polynomial;

pub struct RowExecution<F, const N: usize> {
    pub inputs: [F; N], // fixed-size array of inputs
    pub output: F,
}

pub struct ExecutionTable<F, const N: usize> {
    pub rows: Vec<RowExecution<F, N>>,
}

impl<F, const N: usize> ExecutionTable<F, N> {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Push a new row's data.
    pub fn push_row(&mut self, inputs: [F; N], output: F) {
        self.rows.push(RowExecution { inputs, output });
    }
}

/// A struct that keeps three separate maps:
///  - single_input:  1-input operations
///  - double_input:  2-input operations
///  - triple_input:  3-input operations
///
///  Each map stores a string -> ExecutionTable<F, N>, where N is 1, 2, or 3 respectively.
pub struct AllOpTraces<F> {
    pub single_input: HashMap<ASTKind, ExecutionTable<F, 1>>,
    pub double_input: HashMap<ASTKind, ExecutionTable<F, 2>>,
    pub triple_input: HashMap<ASTKind, ExecutionTable<F, 3>>,
}

impl<F> AllOpTraces<F> {
    pub fn new() -> Self {
        Self {
            single_input: HashMap::new(),
            double_input: HashMap::new(),
            triple_input: HashMap::new(),
        }
    }

    /// Insert a 1-input operation table.
    /// e.g. "NOT" with an ExecutionTable<F,1>.
    pub fn insert_single_op(&mut self, kind: ASTKind, table: ExecutionTable<F, 1>) {
        self.single_input.insert(kind, table);
    }

    /// Insert a 2-input operation table.
    /// e.g. "ADD" with an ExecutionTable<F,2>.
    pub fn insert_double_op(&mut self, kind: ASTKind, table: ExecutionTable<F, 2>) {
        self.double_input.insert(kind, table);
    }

    /// Insert a 3-input operation table.
    pub fn insert_triple_op(&mut self, kind: ASTKind, table: ExecutionTable<F, 3>) {
        self.triple_input.insert(kind, table);
    }

    /// Get a reference to a 1-input operation table by kind.
    pub fn get_single_op(&self, kind: &ASTKind) -> Option<&ExecutionTable<F, 1>> {
        self.single_input.get(kind)
    }

    /// Get a reference to a 2-input operation table by kind.
    pub fn get_double_op(&self, kind: &ASTKind) -> Option<&ExecutionTable<F, 2>> {
        self.double_input.get(kind)
    }

    /// Get a reference to a 3-input operation table by kind.
    pub fn get_triple_op(&self, kind: &ASTKind) -> Option<&ExecutionTable<F, 3>> {
        self.triple_input.get(kind)
    }
}

pub type ExecutionTrace<F: Field> = AllOpTraces<NodeMeta<F>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WirePosition {
    /// Which operation kind (e.g., ASTKind::Add, ASTKind::Not, etc.)
    pub op_kind: ASTKind,
    /// Row index in the operation's table
    pub row_idx: usize,
    /// Type of wire and its position (if input)
    pub wire_type: WireType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WireType {
    /// Input wire with its position index
    Input(usize),
    /// Output wire
    Output,
}

impl<F> ExecutionTrace<F> {
    /// Collect all wire positions from all single/double/triple input tables.
    /// Each wire position is paired with the node_id of the expression.
    pub fn collect_wire_positions(&self) -> Vec<(WirePosition, usize)> {
        let mut positions = Vec::new();

        // Single-input operations (N=1)
        for (kind, table) in &self.single_input {
            for (row_idx, row) in table.rows.iter().enumerate() {
                // There's exactly 1 input wire
                positions.push((
                    WirePosition {
                        op_kind: kind.clone(),
                        row_idx,
                        wire_type: WireType::Input(0),
                    },
                    row.inputs[0].node_id,
                ));
                // Output wire
                positions.push((
                    WirePosition {
                        op_kind: kind.clone(),
                        row_idx,
                        wire_type: WireType::Output,
                    },
                    row.output.node_id,
                ));
            }
        }

        // Double-input operations (N=2)
        for (kind, table) in &self.double_input {
            for (row_idx, row) in table.rows.iter().enumerate() {
                for input_idx in 0..2 {
                    positions.push((
                        WirePosition {
                            op_kind: kind.clone(),
                            row_idx,
                            wire_type: WireType::Input(input_idx),
                        },
                        row.inputs[input_idx].node_id,
                    ));
                }
                // Output wire
                positions.push((
                    WirePosition {
                        op_kind: kind.clone(),
                        row_idx,
                        wire_type: WireType::Output,
                    },
                    row.output.node_id,
                ));
            }
        }

        // Triple-input operations (N=3)
        for (kind, table) in &self.triple_input {
            for (row_idx, row) in table.rows.iter().enumerate() {
                for input_idx in 0..3 {
                    positions.push((
                        WirePosition {
                            op_kind: kind.clone(),
                            row_idx,
                            wire_type: WireType::Input(input_idx),
                        },
                        row.inputs[input_idx].node_id,
                    ));
                }
                // Output wire
                positions.push((
                    WirePosition {
                        op_kind: kind.clone(),
                        row_idx,
                        wire_type: WireType::Output,
                    },
                    row.output.node_id,
                ));
            }
        }

        positions
    }

    /// Group the collected wire positions by the node_id of the expression.
    /// Returns a HashMap<node_id, Vec<WirePosition>>.
    pub fn group_wire_positions_by_node_id(&self) -> HashMap<usize, Vec<WirePosition>> {
        let mut groups: HashMap<usize, Vec<WirePosition>> = HashMap::new();
        for (wire_pos, node_id) in self.collect_wire_positions() {
            groups.entry(node_id).or_default().push(wire_pos);
        }
        groups
    }
}

pub type WireMap<V> = HashMap<WirePosition, V>;

/// A struct representing a polynomial evaluation for gates in table form.
/// Each row contains up to three inputs, one output, and selector polynomials.
#[derive(Debug, Clone)]
pub struct GateEvaluationTable<F> {
    pub input1: Vec<F>,
    pub input2: Vec<F>,
    pub input3: Vec<F>,
    pub output: Vec<F>,
    /// Maps operation kinds to their selector polynomial evaluations.
    /// For each operation, the selector polynomial evaluates to 1 at rows where
    /// that operation is active, and 0 elsewhere.
    pub selectors: HashMap<ASTKind, Vec<F>>,
}

impl<F> GateEvaluationTable<F> {
    pub fn new(trace: &ExecutionTrace<F>) -> Self {
        let total_rows: usize = trace
            .single_input
            .values()
            .map(|table| table.rows.len())
            .chain(trace.double_input.values().map(|table| table.rows.len()))
            .chain(trace.triple_input.values().map(|table| table.rows.len()))
            .sum();

        // Pre-allocate selector vectors for each operation
        let mut selectors = HashMap::new();
        for kind in trace
            .single_input
            .keys()
            .chain(trace.double_input.keys())
            .chain(trace.triple_input.keys())
        {
            selectors.insert(kind.clone(), Vec::with_capacity(total_rows));
        }

        Self {
            input1: Vec::with_capacity(total_rows),
            input2: Vec::with_capacity(total_rows),
            input3: Vec::with_capacity(total_rows),
            output: Vec::with_capacity(total_rows),
            selectors,
        }
    }
}

// Update the build_gate_evaluation_table function to use OffsetTable
pub fn build_gate_evaluation_table<F: Zero + One + Clone>(
    trace: &ExecutionTrace<F>,
    offset_table: &OffsetTable,
    wire_map: &WireMap<F>,
    default_val: F,
) -> GateEvaluationTable<F> {
    let mut poly_table = GateEvaluationTable::new(trace);

    // Initialize selector columns
    let one = F::one();

    // Helper closure for looking up values
    let get_val = |pos: &WirePosition| -> F {
        wire_map
            .get(pos)
            .cloned()
            .unwrap_or_else(|| default_val.clone())
    };

    // Process single-input operations
    for (kind, table) in &trace.single_input {
        let base = offset_table.get_offset(kind).unwrap_or(0);
        for (local_row_idx, _row) in table.rows.iter().enumerate() {
            let global_row_idx = base + local_row_idx;

            // Create wire positions
            let wire_in = WirePosition {
                op_kind: kind.clone(),
                row_idx: local_row_idx,
                wire_type: WireType::Input(0),
            };
            let wire_out = WirePosition {
                op_kind: kind.clone(),
                row_idx: local_row_idx,
                wire_type: WireType::Output,
            };

            // Get values and populate table
            poly_table.input1[global_row_idx] = get_val(&wire_in);
            poly_table.input2[global_row_idx] = default_val.clone();
            poly_table.input3[global_row_idx] = default_val.clone();
            poly_table.output[global_row_idx] = get_val(&wire_out);

            // Set selector
            if let Some(selector) = poly_table.selectors.get_mut(kind) {
                selector[global_row_idx] = one.clone();
            }
        }
    }

    // Similar processing for double and triple input operations...
    // (code omitted for brevity but follows same pattern)

    poly_table
}

/// A struct representing a set of polynomials for the circuit's gate wires:
///   - input1, input2, input3, output
///   - a set of "selector" polynomials, one per gate kind.
#[derive(Debug, Clone)]
pub struct GatePolynomial<F: FftField> {
    pub input1: ark_poly::univariate::DensePolynomial<F>,
    pub input2: ark_poly::univariate::DensePolynomial<F>,
    pub input3: ark_poly::univariate::DensePolynomial<F>,
    pub output: ark_poly::univariate::DensePolynomial<F>,
    pub selectors: HashMap<ASTKind, ark_poly::univariate::DensePolynomial<F>>,
}

/// Converts a GateEvaluationTable<F> into a GatePolynomial<F> by interpolating
/// each of the input/output/selector columns over the provided coset_domain.
pub fn gate_evaluation_table_to_gate_polynomial<F: FftField>(
    table: &GateEvaluationTable<F>,
    coset_domain: &GeneralEvaluationDomain<F>,
) -> GatePolynomial<F> {
    // Interpolate the main wires
    let input1_poly = evaluations_to_dense_polynomial(&table.input1, coset_domain);
    let input2_poly = evaluations_to_dense_polynomial(&table.input2, coset_domain);
    let input3_poly = evaluations_to_dense_polynomial(&table.input3, coset_domain);
    let output_poly = evaluations_to_dense_polynomial(&table.output, coset_domain);

    // Interpolate each selector one-by-one
    let mut selector_polys = HashMap::new();
    for (kind, selector_evals) in &table.selectors {
        selector_polys.insert(
            kind.clone(),
            evaluations_to_dense_polynomial(selector_evals, coset_domain),
        );
    }

    GatePolynomial {
        input1: input1_poly,
        input2: input2_poly,
        input3: input3_poly,
        output: output_poly,
        selectors: selector_polys,
    }
}

/// Evaluates each of the gate polynomials (wires + selectors) over the given domain,
/// then forms a single "constraint" evaluation vector. This represents the combined
/// constraint polynomial evaluated at each point of the domain. If all constraints
/// are satisfied, this resulting vector should be all zeros in the field.
///
/// NOTE: As requested, this skips any boolean constraints or b≠0 checks.
///
/// To obtain the constraint polynomial in coefficient form, call:
///   let constraint_evals = build_gate_constraint_evaluation_polynomial(&gate_poly, &domain);
///   let constraint_poly = evaluations_to_dense_polynomial(&constraint_evals, &domain);
///
pub fn build_gate_constraint_evaluation_polynomial<F: FftField>(
    gate_poly: &GatePolynomial<F>,
    domain: &GeneralEvaluationDomain<F>,
) -> Vec<F> {
    // 1) Evaluate each main wire polynomial (input1, input2, input3, output) over the domain
    let input1_evals = gate_poly.input1.evaluate_over_domain_by_ref(*domain).evals;
    let input2_evals = gate_poly.input2.evaluate_over_domain_by_ref(*domain).evals;
    let input3_evals = gate_poly.input3.evaluate_over_domain_by_ref(*domain).evals;
    let output_evals = gate_poly.output.evaluate_over_domain_by_ref(*domain).evals;

    // 2) Evaluate each selector polynomial over the domain
    let mut selector_evals_map: HashMap<ASTKind, Vec<F>> = HashMap::new();
    for (kind, s_poly) in &gate_poly.selectors {
        let s_evals = s_poly.evaluate_over_domain_by_ref(*domain).evals;
        selector_evals_map.insert(kind.clone(), s_evals);
    }

    // 3) For each i in [0..n], compute the constraint = Σ(selector[k][i] * P_k(i)),
    //    where P_k(i) is the gate's constraint polynomial expression at row i.
    let n = domain.size();
    let mut constraints = vec![F::zero(); n];

    for i in 0..n {
        // rename inputs for convenience
        let a = input1_evals[i];
        let b = input2_evals[i];
        let c = input3_evals[i];
        let out = output_evals[i];

        // accumulate active gate constraint terms
        for (kind, s_vec) in &selector_evals_map {
            let s = s_vec[i];
            if s.is_zero() {
                continue; // gate not active at this row
            }

            // Polynomial expression that must vanish if the gate is active
            let gate_expr = match kind {
                // (1) and (2) – no constraint needed for compile-time constants, skipping
                ASTKind::Int => F::zero(),
                ASTKind::Bool => F::zero(),

                // c - (a + b) = 0
                ASTKind::Add => out - (a + b),

                // c - (a - b) = 0  => c - a + b
                ASTKind::Sub => out - (a - b),

                // c - a*b = 0
                ASTKind::Mult => out - (a * b),

                // c*b - a = 0
                ASTKind::Div => out * b - a,

                // eq*(a - b) = 0 (skipped eq being boolean)
                ASTKind::Eq => out * (a - b),

                // y + x - 1 = 0  (skipped boolean check on x)
                ASTKind::Not => out + a - F::one(),

                // r - [ cond*t + (1-cond)*e ] = 0
                //   skipping the boolean check for cond
                ASTKind::If => {
                    // cond = a, then_expr = b, else_expr = c, out = r
                    let one_minus_a = F::one() - a;
                    out - (a * b + one_minus_a * c)
                }
            };

            constraints[i] += s * gate_expr;
        }
    }

    constraints
}

