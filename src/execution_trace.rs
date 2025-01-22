use std::collections::HashMap;

use ark_ff::{Field, One, Zero};

use crate::ast::NodeMeta;

pub struct RowExecution<F, const N: usize> {
    pub inputs: [F; N],  // fixed-size array of inputs
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
    pub single_input: HashMap<String, ExecutionTable<F, 1>>,
    pub double_input: HashMap<String, ExecutionTable<F, 2>>,
    pub triple_input: HashMap<String, ExecutionTable<F, 3>>,
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
    pub fn insert_single_op(&mut self, op_name: &str, table: ExecutionTable<F, 1>) {
        self.single_input.insert(op_name.to_string(), table);
    }

    /// Insert a 2-input operation table.
    /// e.g. "ADD" with an ExecutionTable<F,2>.
    pub fn insert_double_op(&mut self, op_name: &str, table: ExecutionTable<F, 2>) {
        self.double_input.insert(op_name.to_string(), table);
    }

    /// Insert a 3-input operation table.
    pub fn insert_triple_op(&mut self, op_name: &str, table: ExecutionTable<F, 3>) {
        self.triple_input.insert(op_name.to_string(), table);
    }

    /// Get a reference to a 1-input operation table by name.
    pub fn get_single_op(&self, op_name: &str) -> Option<&ExecutionTable<F, 1>> {
        self.single_input.get(op_name)
    }

    /// Get a reference to a 2-input operation table by name.
    pub fn get_double_op(&self, op_name: &str) -> Option<&ExecutionTable<F, 2>> {
        self.double_input.get(op_name)
    }

    /// Get a reference to a 3-input operation table by name.
    pub fn get_triple_op(&self, op_name: &str) -> Option<&ExecutionTable<F, 3>> {
        self.triple_input.get(op_name)
    }
}

pub type ExecutionTrace<F: Field> = AllOpTraces<NodeMeta<F>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WirePosition {
    /// Name of the operation (e.g., "ADD", "MUL", etc.)
    pub op_name: String,
    /// Row index in the operation's table
    pub row_idx: usize,
    /// Type of wire and its position (if input)
    pub wire_type: WireType
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WireType {
    /// Input wire with its position index
    Input(usize),
    /// Output wire
    Output,
}

pub type PermutationTrace = AllOpTraces<WirePosition>;

impl<F: Field> ExecutionTrace<F> {
    /// Collect all wire positions from all single/double/triple input tables.
    /// Each wire position is paired with the node_id of the expression.
    pub fn collect_wire_positions(&self) -> Vec<(WirePosition, usize)> {
        let mut positions = Vec::new();

        // Single-input operations (N=1)
        for (op_name, table) in &self.single_input {
            for (row_idx, row) in table.rows.iter().enumerate() {
                // There's exactly 1 input wire
                positions.push((
                    WirePosition {
                        op_name: op_name.clone(),
                        row_idx,
                        wire_type: WireType::Input(0),
                    },
                    row.inputs[0].node_id
                ));
                // Output wire
                positions.push((
                    WirePosition {
                        op_name: op_name.clone(),
                        row_idx,
                        wire_type: WireType::Output,
                    },
                    row.output.node_id
                ));
            }
        }

        // Double-input operations (N=2)
        for (op_name, table) in &self.double_input {
            for (row_idx, row) in table.rows.iter().enumerate() {
                for input_idx in 0..2 {
                    positions.push((
                        WirePosition {
                            op_name: op_name.clone(),
                            row_idx,
                            wire_type: WireType::Input(input_idx),
                        },
                        row.inputs[input_idx].node_id
                    ));
                }
                // Output wire
                positions.push((
                    WirePosition {
                        op_name: op_name.clone(),
                        row_idx,
                        wire_type: WireType::Output,
                    },
                    row.output.node_id
                ));
            }
        }

        // Triple-input operations (N=3)
        for (op_name, table) in &self.triple_input {
            for (row_idx, row) in table.rows.iter().enumerate() {
                for input_idx in 0..3 {
                    positions.push((
                        WirePosition {
                            op_name: op_name.clone(),
                            row_idx,
                            wire_type: WireType::Input(input_idx),
                        },
                        row.inputs[input_idx].node_id
                    ));
                }
                // Output wire
                positions.push((
                    WirePosition {
                        op_name: op_name.clone(),
                        row_idx,
                        wire_type: WireType::Output,
                    },
                    row.output.node_id
                ));
            }
        }

        positions
    }

    /// Group the collected wire positions by the node_id of the expression.
    /// Returns a HashMap<node_id, Vec<WirePosition>>.
    pub fn group_wire_positions_by_node_id(
        &self
    ) -> HashMap<usize, Vec<WirePosition>> {
        let mut groups: HashMap<usize, Vec<WirePosition>> = HashMap::new();
        for (wire_pos, node_id) in self.collect_wire_positions() {
            groups.entry(node_id).or_default().push(wire_pos);
        }
        groups
    }
}

type WireMap<V> = HashMap<WirePosition, V>;

type FlatPermutationMap = WireMap<WirePosition>;


/// Build a permutation trace (sigma) by linking each group of wire positions
/// that share the same node_id in a cycle. For example, if node_id=17 appears in
/// four distinct wire positions [p0, p1, p2, p3], we arrange them in a cycle:
/// p0→p1, p1→p2, p2→p3, p3→p0.
pub fn build_permutation_trace<F: Field>(trace: &ExecutionTrace<F>) -> FlatPermutationMap {
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

/// A struct representing a polynomial evaluation table for gates.
/// Each row contains up to three inputs, one output, and selector polynomials.
#[derive(Debug, Clone)]
pub struct PolynomialEvaluationTable<F> {
    pub input1: Vec<F>,
    pub input2: Vec<F>,
    pub input3: Vec<F>,
    pub output: Vec<F>,
    /// Maps operation names to their selector polynomial evaluations.
    /// For each operation, the selector polynomial evaluates to 1 at rows where
    /// that operation is active, and 0 elsewhere.
    pub selectors: HashMap<String, Vec<F>>
}
impl<F> PolynomialEvaluationTable<F> {
    pub fn new(trace: &ExecutionTrace<F>) -> Self {
        let total_rows: usize = trace.single_input.values()
            .map(|table| table.rows.len())
            .chain(trace.double_input.values().map(|table| table.rows.len()))
            .chain(trace.triple_input.values().map(|table| table.rows.len()))
            .sum();

        // Pre-allocate selector vectors for each operation
        let mut selectors = HashMap::new();
        for op_name in trace.single_input.keys()
            .chain(trace.double_input.keys())
            .chain(trace.triple_input.keys()) {
            selectors.insert(op_name.clone(), Vec::with_capacity(total_rows));
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

/// A struct that holds both the offset map and total size of all tables.
/// This helps track where each operation's rows begin in the flattened table
/// and the total number of rows across all operations.
#[derive(Debug, Clone)]
pub struct OffsetTable {
    /// Maps operation names to their starting row index
    pub offset_map: HashMap<String, usize>,
    /// Total number of rows across all operations
    pub total_rows: usize,
}

impl OffsetTable {
    /// Builds an OffsetTable from an ExecutionTrace by computing
    /// cumulative row counts across all operations (single/double/triple).
    pub fn build<F: Field>(trace: &ExecutionTrace<F>) -> Self {
        let mut offset_map = HashMap::new();
        let mut cumulative = 0;

        // Single-input operations
        for (op_name, table) in &trace.single_input {
            offset_map.insert(op_name.clone(), cumulative);
            cumulative += table.rows.len();
        }

        // Double-input operations
        for (op_name, table) in &trace.double_input {
            offset_map.insert(op_name.clone(), cumulative);
            cumulative += table.rows.len();
        }

        // Triple-input operations
        for (op_name, table) in &trace.triple_input {
            offset_map.insert(op_name.clone(), cumulative);
            cumulative += table.rows.len();
        }

        Self {
            offset_map,
            total_rows: cumulative,
        }
    }

    /// Get the starting row offset for an operation
    pub fn get_offset(&self, op_name: &str) -> Option<usize> {
        self.offset_map.get(op_name).copied()
    }

    /// Convert a local row index within an operation to a global row index
    pub fn to_global_row(&self, op_name: &str, local_row: usize) -> Option<usize> {
        self.get_offset(op_name).map(|offset| offset + local_row)
    }

    /// Convert a WirePosition to a global index in polynomials
    pub fn to_index(&self, pos: &WirePosition) -> usize {
        let base = self.get_offset(&pos.op_name).unwrap_or(0);
        base + pos.row_idx
    }
}


// Update the build_polynomial_evaluation_table function to use OffsetTable
pub fn build_polynomial_evaluation_table<F: Zero + One + Clone>(
    trace: &ExecutionTrace<F>,
    offset_table: &OffsetTable,
    wire_map: &WireMap<F>,
    default_val: F,
) -> PolynomialEvaluationTable<F> {
    let mut poly_table = PolynomialEvaluationTable::new(trace);

    // Initialize selector columns
    let one = F::one();

    // Helper closure for looking up values
    let get_val = |pos: &WirePosition| -> F {
        wire_map.get(pos).cloned().unwrap_or_else(|| default_val.clone())
    };

    // Process single-input operations
    for (op_name, table) in &trace.single_input {
        let base = offset_table.get_offset(op_name).unwrap_or(0);
        for (local_row_idx, row) in table.rows.iter().enumerate() {
            let global_row_idx = base + local_row_idx;
            
            // Create wire positions
            let wire_in = WirePosition {
                op_name: op_name.clone(),
                row_idx: local_row_idx,
                wire_type: WireType::Input(0)
            };
            let wire_out = WirePosition {
                op_name: op_name.clone(),
                row_idx: local_row_idx,
                wire_type: WireType::Output
            };

            // Get values and populate table
            poly_table.input1[global_row_idx] = get_val(&wire_in);
            poly_table.input2[global_row_idx] = default_val.clone();
            poly_table.input3[global_row_idx] = default_val.clone();
            poly_table.output[global_row_idx] = get_val(&wire_out);

            // Set selector
            if let Some(selector) = poly_table.selectors.get_mut(op_name) {
                selector[global_row_idx] = one.clone();
            }
        }
    }

    // Similar processing for double and triple input operations...
    // (code omitted for brevity but follows same pattern)

    poly_table
}

/// A struct representing the permutation polynomials for each wire.
/// Each vector represents σ₁, σ₂, σ₃, σ₄ in Plonk terminology.
#[derive(Debug, Clone)]
pub struct PermutationPolynomials<F> {
    /// σ₁: permutation for first input wire
    pub sigma1: Vec<F>,
    /// σ₂: permutation for second input wire
    pub sigma2: Vec<F>,
    /// σ₃: permutation for third input wire
    pub sigma3: Vec<F>,
    /// σ₄: permutation for output wire
    pub sigma4: Vec<F>,
}

impl<F> PermutationPolynomials<F> {
    pub fn new(size: usize, default: F) -> Self 
    where 
        F: Clone,
    {
        Self {
            sigma1: vec![default.clone(); size],
            sigma2: vec![default.clone(); size],
            sigma3: vec![default.clone(); size],
            sigma4: vec![default.clone(); size],
        }
    }
}

/// Build permutation polynomials from a flat permutation map.
/// For each position in the circuit, compute its "next" position in the permutation cycle.
pub fn build_permutation_polynomials<F: Field>(
    trace: &ExecutionTrace<F>,
    offset_table: &OffsetTable,
    perm_map: &FlatPermutationMap,
) -> PermutationPolynomials<F> 
where
    F: Clone + From<usize> + Zero,
{
    let mut poly = PermutationPolynomials::new(offset_table.total_rows, F::zero());

    // Helper function to get the next index in the permutation
    let get_next_index = |wire: &WirePosition| -> usize {
        let curr_idx = offset_table.to_index(wire);
        perm_map.get(wire)
            .map(|next| offset_table.to_index(next))
            .unwrap_or(curr_idx)
    };

    // Process single-input operations
    for (op_name, table) in &trace.single_input {
        for (row_idx, _row) in table.rows.iter().enumerate() {
            // Current wire positions
            let wire_in = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Input(0),
            };
            let wire_out = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Output,
            };

            // Get next positions in permutation
            let curr_in_idx = offset_table.to_index(&wire_in);
            poly.sigma1[curr_in_idx] = F::from(get_next_index(&wire_in));

            let curr_out_idx = offset_table.to_index(&wire_out);
            poly.sigma4[curr_out_idx] = F::from(get_next_index(&wire_out));

            // For unused wires in single-input ops, point to self
            poly.sigma2[curr_in_idx] = F::from(curr_in_idx);
            poly.sigma3[curr_in_idx] = F::from(curr_in_idx);
        }
    }

    // Process double-input operations
    for (op_name, table) in &trace.double_input {
        for (row_idx, _row) in table.rows.iter().enumerate() {
            // Current wire positions
            let wire_in0 = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Input(0),
            };
            let wire_in1 = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Input(1),
            };
            let wire_out = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Output,
            };

            // Get next positions in permutation
            let curr_in0_idx = offset_table.to_index(&wire_in0);
            poly.sigma1[curr_in0_idx] = F::from(get_next_index(&wire_in0));

            let curr_in1_idx = offset_table.to_index(&wire_in1);
            poly.sigma2[curr_in1_idx] = F::from(get_next_index(&wire_in1));

            let curr_out_idx = offset_table.to_index(&wire_out);
            poly.sigma4[curr_out_idx] = F::from(get_next_index(&wire_out));

            // For unused wire in double-input ops, point to self
            poly.sigma3[curr_in0_idx] = F::from(curr_in0_idx);
        }
    }

    // Process triple-input operations
    for (op_name, table) in &trace.triple_input {
        for (row_idx, _row) in table.rows.iter().enumerate() {
            // Current wire positions
            let wire_in0 = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Input(0),
            };
            let wire_in1 = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Input(1),
            };
            let wire_in2 = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Input(2),
            };
            let wire_out = WirePosition {
                op_name: op_name.clone(),
                row_idx,
                wire_type: WireType::Output,
            };

            // Get next positions in permutation
            let curr_in0_idx = offset_table.to_index(&wire_in0);
            poly.sigma1[curr_in0_idx] = F::from(get_next_index(&wire_in0));

            let curr_in1_idx = offset_table.to_index(&wire_in1);
            poly.sigma2[curr_in1_idx] = F::from(get_next_index(&wire_in1));

            let curr_in2_idx = offset_table.to_index(&wire_in2);
            poly.sigma3[curr_in2_idx] = F::from(get_next_index(&wire_in2));

            let curr_out_idx = offset_table.to_index(&wire_out);
            poly.sigma4[curr_out_idx] = F::from(get_next_index(&wire_out));
        }
    }

    poly
}

// This function builds the accumulation polynomial combining the wire permutation polynomials.
// It takes the permutation polynomials and compute the recursive relation for the accumulation polynomial.
// Namely, expression of this polynomial is Z(w^(i + 1)) = Z(w^i) * (w^i + bind_offset) + bind_multiplier_constant.
// pub fn build_accumulation_polynomial<F>(
//     perm_polys: &PermutationPolynomials<F>,
//     wire_map: &WireMap<F>,
//     bind_offset: usize,
//     bind_multiplier_constant: usize,
// ) -> Polynomial<F> {
//     todo!()
// }