use std::collections::HashMap;

use ark_ff::{Field, One, Zero};

use crate::ast::NodeMeta;
use crate::offset_table::OffsetTable;

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

impl<F> ExecutionTrace<F> {
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

pub type WireMap<V> = HashMap<WirePosition, V>;


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
