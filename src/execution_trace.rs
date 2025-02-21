use std::collections::{HashMap, HashSet};

use ark_ff::{Field, One, Zero};
use ark_poly::{GeneralEvaluationDomain, Polynomial};

use crate::plonk_circuit::{EvalValue, NodeMeta, PlonkNode, PlonkNodeId, PlonkNodeKind};
use crate::union_find::UnionFind;

pub struct ExecutionCell<F> {
    node_id: Option<PlonkNodeId>,
    value: F,
}

pub struct ExecutionRow<F, const N: usize> {
    pub operation: PlonkNodeKind,
    pub inputs: [ExecutionCell<F>; N], // fixed-size array of inputs
    pub output: ExecutionCell<F>,
}

pub struct PlonkContraints<F> {
    pub gate_operations: Vec<ExecutionRow<F, 3>>,
    pub node_cell_equivalences: Vec<(PlonkNodeId, PositionCell)>, // This tells us if a plonk node expression is equivalent to a wire cell
    pub node_node_equivalences: UnionFind<PlonkNodeId>,           // Now using the generic UnionFind
}

impl<F> PlonkContraints<F> {
    pub fn new(equivalence_pairs: Vec<(PlonkNodeId, PlonkNodeId)>) -> Self {
        let mut node_node_equivalences = UnionFind::new();
        node_node_equivalences.add_equivalences(&equivalence_pairs);

        Self {
            gate_operations: Vec::new(),
            node_cell_equivalences: Vec::new(),
            node_node_equivalences,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ColumnType {
    Input(i32),
    Output,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PositionCell {
    pub row_idx: usize,
    pub wire_type: ColumnType,
}

// This type demonstrates if groups of cells are equivalent to each other
pub type WireCellEquivalences = HashMap<PlonkNodeId, HashSet<PositionCell>>;

// This is a mapping from one wire cell to another wire cell
pub type PermutationMap = HashMap<PositionCell, PositionCell>;

/// A struct representing a polynomial evaluation for gates in table form.
/// Each row contains up to three inputs, one output, and selector polynomials.
#[derive(Debug, Clone)]
pub struct ExecutionTraceTable<F> {
    pub input1: Vec<F>,
    pub input2: Vec<F>,
    pub input3: Vec<F>,
    pub output: Vec<F>,

    // This is the permutation mapping of the input wires
    pub permutation_input1: Vec<F>,
    pub permutation_input2: Vec<F>,
    pub permutation_input3: Vec<F>,
    pub permutation_output: Vec<F>,

    /// Maps operation kinds to their selector polynomial evaluations.
    /// For each operation, the selector polynomial evaluates to 1 at rows where
    /// that operation is active, and 0 elsewhere.
    pub selectors: HashMap<PlonkNodeKind, Vec<F>>,
}

/// This helper function is intended to interpret a `PlonkNode<NodeMeta<EvalValue>>`
/// and adds the constraints of each node into a table.
/// It will likely gather information about inputs, outputs, and any intermediate
/// nodes, then populate rows in the execution trace table. Currently unimplemented.
fn interpret_plonk_node_to_execution_trace_table_helper<F: Field>(
    node: &PlonkNode<NodeMeta<F>>,
    constraints: &mut PlonkContraints<F>,
) {
    match node {
        PlonkNode::Int(value, meta) => {
            // For constants, we add a constraint that the output equals the constant
            let row = ExecutionRow {
                operation: PlonkNodeKind::Int,
                inputs: [
                    ExecutionCell {
                        node_id: Some(meta.node_id),
                        value: F::from(*value as u64),
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: F::from(*value as u64),
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::Bool(value, meta) => {
            // For boolean constants, similar to Int but with 0/1 values
            let f_value = if *value { F::one() } else { F::zero() };
            let row = ExecutionRow {
                operation: PlonkNodeKind::Bool,
                inputs: [
                    ExecutionCell {
                        node_id: Some(meta.node_id),
                        value: f_value,
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: f_value,
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::Add(lhs, rhs, meta) => {
            // Process child nodes first
            interpret_plonk_node_to_execution_trace_table_helper(lhs, constraints);
            interpret_plonk_node_to_execution_trace_table_helper(rhs, constraints);

            // Add the addition constraint
            let row = ExecutionRow {
                operation: PlonkNodeKind::Add,
                inputs: [
                    ExecutionCell {
                        node_id: Some(lhs.meta().node_id),
                        value: lhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: Some(rhs.meta().node_id),
                        value: rhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: meta.evaluated_value,
                },
            };
            constraints.gate_operations.push(row);

            // Add node equivalences
            constraints.node_cell_equivalences.push((
                lhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(0),
                },
            ));
            constraints.node_cell_equivalences.push((
                rhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(1),
                },
            ));
            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::Sub(lhs, rhs, meta) => {
            interpret_plonk_node_to_execution_trace_table_helper(lhs, constraints);
            interpret_plonk_node_to_execution_trace_table_helper(rhs, constraints);

            let row = ExecutionRow {
                operation: PlonkNodeKind::Sub,
                inputs: [
                    ExecutionCell {
                        node_id: Some(lhs.meta().node_id),
                        value: lhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: Some(rhs.meta().node_id),
                        value: rhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: meta.evaluated_value,
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                lhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(0),
                },
            ));

            constraints.node_cell_equivalences.push((
                rhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(1),
                },
            ));
            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::Mult(lhs, rhs, meta) => {
            interpret_plonk_node_to_execution_trace_table_helper(lhs, constraints);
            interpret_plonk_node_to_execution_trace_table_helper(rhs, constraints);

            let row = ExecutionRow {
                operation: PlonkNodeKind::Mult,
                inputs: [
                    ExecutionCell {
                        node_id: Some(lhs.meta().node_id),
                        value: lhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: Some(rhs.meta().node_id),
                        value: rhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: meta.evaluated_value,
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                lhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(0),
                },
            ));
            constraints.node_cell_equivalences.push((
                rhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(1),
                },
            ));
            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::Div(lhs, rhs, meta) => {
            interpret_plonk_node_to_execution_trace_table_helper(lhs, constraints);
            interpret_plonk_node_to_execution_trace_table_helper(rhs, constraints);

            let row = ExecutionRow {
                operation: PlonkNodeKind::Div,
                inputs: [
                    ExecutionCell {
                        node_id: Some(lhs.meta().node_id),
                        value: lhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: Some(rhs.meta().node_id),
                        value: rhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: meta.evaluated_value,
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                lhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(0),
                },
            ));

            constraints.node_cell_equivalences.push((
                rhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(1),
                },
            ));
            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::Eq(lhs, rhs, meta) => {
            interpret_plonk_node_to_execution_trace_table_helper(lhs, constraints);
            interpret_plonk_node_to_execution_trace_table_helper(rhs, constraints);

            let row = ExecutionRow {
                operation: PlonkNodeKind::Eq,
                inputs: [
                    ExecutionCell {
                        node_id: Some(lhs.meta().node_id),
                        value: lhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: Some(rhs.meta().node_id),
                        value: rhs.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: meta.evaluated_value,
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                lhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(0),
                },
            ));
            constraints.node_cell_equivalences.push((
                rhs.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(1),
                },
            ));
            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::Not(sub, meta) => {
            interpret_plonk_node_to_execution_trace_table_helper(sub, constraints);

            let row = ExecutionRow {
                operation: PlonkNodeKind::Not,
                inputs: [
                    ExecutionCell {
                        node_id: Some(sub.meta().node_id),
                        value: sub.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                    ExecutionCell {
                        node_id: None,
                        value: F::zero(),
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: meta.evaluated_value,
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                sub.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(0),
                },
            ));

            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        PlonkNode::If(cond, then_branch, else_branch, meta) => {
            interpret_plonk_node_to_execution_trace_table_helper(cond, constraints);
            interpret_plonk_node_to_execution_trace_table_helper(then_branch, constraints);
            interpret_plonk_node_to_execution_trace_table_helper(else_branch, constraints);

            let row = ExecutionRow {
                operation: PlonkNodeKind::If,
                inputs: [
                    ExecutionCell {
                        node_id: Some(cond.meta().node_id),
                        value: cond.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: Some(then_branch.meta().node_id),
                        value: then_branch.meta().evaluated_value,
                    },
                    ExecutionCell {
                        node_id: Some(else_branch.meta().node_id),
                        value: else_branch.meta().evaluated_value,
                    },
                ],
                output: ExecutionCell {
                    node_id: Some(meta.node_id),
                    value: meta.evaluated_value,
                },
            };
            constraints.gate_operations.push(row);

            constraints.node_cell_equivalences.push((
                cond.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(0),
                },
            ));
            constraints.node_cell_equivalences.push((
                then_branch.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(1),
                },
            ));
            constraints.node_cell_equivalences.push((
                else_branch.meta().node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Input(2),
                },
            ));

            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        // TODO: Figure out
        PlonkNode::Let(_var_name, bound_expr, body_expr, meta) => {
            // Process the bound expression
            interpret_plonk_node_to_execution_trace_table_helper(bound_expr, constraints);

            // Process the body
            interpret_plonk_node_to_execution_trace_table_helper(body_expr, constraints);

            // Add equivalence between the bound expression and its uses in the body
            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
        // TODO: Figure out
        PlonkNode::Var(_, meta) => {
            // Variables are handled through the node equivalences system
            // Their values are propagated through the wire equivalences
            constraints.node_cell_equivalences.push((
                meta.node_id,
                PositionCell {
                    row_idx: constraints.gate_operations.len() - 1,
                    wire_type: ColumnType::Output,
                },
            ));
        }
    }
}

// Interprets the plonk node to execution trace.
// It converts a node into a ExecutionTraceTable. It first uses interpret_plonk_node_to_execution_trace_table_helper to get the wire constraints and permutation constraints.
// Then, it uses build_permutation_map to build a permutation map.
// Then, you construct a matrix of the permutation and fulfill the values for the ExecutionTraceTable
#[allow(dead_code)]
pub fn interpret_plonk_node_to_execution_trace_table<F: Field>(
    node: &PlonkNode<NodeMeta<F>>,
    node_node_equivalences: Vec<(PlonkNodeId, PlonkNodeId)>,
) -> ExecutionTraceTable<F> {
    // Create a new constraints object
    let mut constraints = PlonkContraints::new(node_node_equivalences);

    // Convert the PlonkNode into constraints
    interpret_plonk_node_to_execution_trace_table_helper(node, &mut constraints);

    // Build the execution trace table from the constraints
    build_execution_trace_table(&mut constraints)
}

// Updated function to take WireCellEquivalences directly
fn build_permutation_map<F: Field>(
    wire_equivalences: &WireCellEquivalences,
    gate_operations: &[ExecutionRow<F, 3>],
) -> PermutationMap {
    let mut permutation_map: PermutationMap = HashMap::new();

    // 1. Process all equivalence groups (including single-element)
    for positions in wire_equivalences.values() {
        let positions: Vec<_> = positions.iter().collect();

        // Create cycle even for single-element groups
        for i in 0..positions.len() {
            let from = positions[i];
            let to = positions[(i + 1) % positions.len()];
            permutation_map.insert(from.clone(), to.clone());
        }
    }

    // 2. Add self-mapping for all cells (including empty ones)
    for (row_idx, row) in gate_operations.iter().enumerate() {
        // Process all input wires (0,1,2) and output wire
        for input_idx in 0..3 {
            let pos = PositionCell {
                row_idx,
                wire_type: ColumnType::Input(input_idx),
            };
            permutation_map
                .entry(pos.clone())
                .or_insert_with(|| pos.clone());
        }

        let output_pos = PositionCell {
            row_idx,
            wire_type: ColumnType::Output,
        };
        permutation_map
            .entry(output_pos.clone())
            .or_insert_with(|| output_pos.clone());
    }

    permutation_map
}

// Update the function to properly use the generic UnionFind interface
fn build_permutation_groups<F: Field>(
    plonk_constraints: &mut PlonkContraints<F>,
) -> WireCellEquivalences {
    let mut wire_equivalences: WireCellEquivalences = HashMap::new();

    // Process all cells in gate operations
    for (row_idx, row) in plonk_constraints.gate_operations.iter().enumerate() {
        // Process inputs
        for (input_idx, cell) in row.inputs.iter().enumerate() {
            if let Some(node_id) = cell.node_id {
                let root = plonk_constraints.node_node_equivalences.find(&node_id);
                let pos = PositionCell {
                    row_idx,
                    wire_type: ColumnType::Input(input_idx as i32),
                };
                wire_equivalences.entry(root).or_default().insert(pos);
            }
        }

        // Process output
        if let Some(node_id) = row.output.node_id {
            let root = plonk_constraints.node_node_equivalences.find(&node_id);
            let pos = PositionCell {
                row_idx,
                wire_type: ColumnType::Output,
            };
            wire_equivalences.entry(root).or_default().insert(pos);
        }
    }

    wire_equivalences
}

/// Helper function to convert a PositionCell to a unique identifier
fn position_to_id(pos: &PositionCell) -> usize {
    let base = pos.row_idx * 4; // 4 columns per row (3 inputs + 1 output)
    match pos.wire_type {
        ColumnType::Input(0) => base,
        ColumnType::Input(1) => base + 1,
        ColumnType::Input(2) => base + 2,
        ColumnType::Output => base + 3,
        _ => panic!("Invalid column type"),
    }
}

pub fn build_execution_trace_table<F: Field>(
    plonk_constraints: &mut PlonkContraints<F>,
) -> ExecutionTraceTable<F> {
    let num_rows = plonk_constraints.gate_operations.len();

    // Initialize vectors for all columns
    let mut input1 = vec![F::zero(); num_rows];
    let mut input2 = vec![F::zero(); num_rows];
    let mut input3 = vec![F::zero(); num_rows];
    let mut output = vec![F::zero(); num_rows];

    // Initialize permutation vectors
    let mut permutation_input1 = vec![F::zero(); num_rows];
    let mut permutation_input2 = vec![F::zero(); num_rows];
    let mut permutation_input3 = vec![F::zero(); num_rows];
    let mut permutation_output = vec![F::zero(); num_rows];

    // Initialize selector polynomials
    let mut selectors: HashMap<PlonkNodeKind, Vec<F>> = HashMap::new();
    for kind in [
        PlonkNodeKind::Int,
        PlonkNodeKind::Bool,
        PlonkNodeKind::Add,
        PlonkNodeKind::Sub,
        PlonkNodeKind::Mult,
        PlonkNodeKind::Div,
        PlonkNodeKind::Eq,
        PlonkNodeKind::Not,
        PlonkNodeKind::If,
    ]
    .iter()
    {
        selectors.insert(kind.clone(), vec![F::zero(); num_rows]);
    }

    // Fill in the values from gate operations
    for (row_idx, gate) in plonk_constraints.gate_operations.iter().enumerate() {
        // Fill in inputs and output
        input1[row_idx] = gate.inputs[0].value;
        input2[row_idx] = gate.inputs[1].value;
        input3[row_idx] = gate.inputs[2].value;
        output[row_idx] = gate.output.value;

        // Set selector polynomial for this operation to 1 at this row
        if let Some(selector) = selectors.get_mut(&gate.operation) {
            selector[row_idx] = F::one();
        }
    }

    // Build permutation map
    let wire_equivalences = build_permutation_groups(plonk_constraints);
    let permutation_map =
        build_permutation_map(&wire_equivalences, &plonk_constraints.gate_operations);

    // Fill in permutation values using unique IDs
    for row_idx in 0..num_rows {
        // Process all wire types for this row
        let wire_types = [
            ColumnType::Input(0),
            ColumnType::Input(1),
            ColumnType::Input(2),
            ColumnType::Output,
        ];

        for wire_type in wire_types {
            let current_pos = PositionCell {
                row_idx,
                wire_type: wire_type.clone(),
            };
            let target_pos = permutation_map.get(&current_pos).unwrap_or(&current_pos);
            let target_id = position_to_id(target_pos);

            match wire_type.clone() {
                ColumnType::Input(0) => permutation_input1[row_idx] = F::from(target_id as u64),
                ColumnType::Input(1) => permutation_input2[row_idx] = F::from(target_id as u64),
                ColumnType::Input(2) => permutation_input3[row_idx] = F::from(target_id as u64),
                ColumnType::Output => permutation_output[row_idx] = F::from(target_id as u64),
                _ => unreachable!(),
            }
        }
    }

    ExecutionTraceTable {
        input1,
        input2,
        input3,
        output,
        permutation_input1,
        permutation_input2,
        permutation_input3,
        permutation_output,
        selectors,
    }
}

impl<F: Field + std::fmt::Debug> std::fmt::Display for PlonkContraints<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PLONK Constraints:")?;
        for gate in self.gate_operations.iter() {
            writeln!(
                f,
                "{:?}, [({:?}, {:?}), ({:?}, {:?}), ({:?}, {:?})], ({:?}, {:?})",
                gate.operation,
                gate.inputs[0].node_id,
                gate.inputs[0].value,
                gate.inputs[1].node_id,
                gate.inputs[1].value,
                gate.inputs[2].node_id,
                gate.inputs[2].value,
                gate.output.node_id,
                gate.output.value,
            )?;
        }

        writeln!(f, "\nPermutation Constraints:")?;
        for (node_id, position) in &self.node_cell_equivalences {
            writeln!(
                f,
                "Node {} -> row {}, {:?}",
                node_id, position.row_idx, position.wire_type
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::ast_to_plonk::convert_to_plonk;
    use crate::execution_trace::{
        build_permutation_map, interpret_plonk_node_to_execution_trace_table_helper, ColumnType,
        ExecutionCell, ExecutionRow, PlonkContraints, PositionCell,
    };
    use crate::plonk_circuit::evaluated_node_factory::{add, int, let_, mult, var};
    use crate::plonk_circuit::{evaluated_node_factory, EvaluatedPlonk, PlonkNodeId};
    use std::collections::{HashMap, HashSet};

    use ark_bn254::Fr as F;
    use ark_ff::Field;

    use super::build_permutation_groups;

    // Wire cell counter using the corrected permutation groups
    pub fn count_wire_cell_references<F: Field>(
        plonk_constraints: &mut PlonkContraints<F>,
        node_ids: &[PlonkNodeId],
    ) -> HashMap<PlonkNodeId, usize> {
        let wire_equivalences = build_permutation_groups(plonk_constraints);

        node_ids.iter().fold(HashMap::new(), |mut acc, &node_id| {
            let root = plonk_constraints.node_node_equivalences.find(&node_id);
            let count = wire_equivalences
                .get(&root)
                .map(|cells| cells.len())
                .unwrap_or(0);
            acc.insert(node_id, count);
            acc
        })
    }

    fn run_permutation_bijection_test(input: &str) {
        use crate::language_frontend::{lexer::lex, parser};
        use crate::plonk_circuit::eval_plonk_node;

        fn parse_expr(input: &str) -> Box<crate::language_frontend::ast::Expr> {
            let tokens = lex(input);
            let token_triples: Vec<_> = tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| (i, t, i + 1))
                .collect();

            parser::ExprParser::new()
                .parse(token_triples.into_iter())
                .unwrap()
        }

        let expr = parse_expr(input);
        let plonk_node = convert_to_plonk(&expr).unwrap();
        let evaluated_plonk = eval_plonk_node::<F>(&plonk_node)
            .unwrap_or_else(|e| panic!("Evaluation failed for '{}': {:?}", input, e));

        let mut constraints = PlonkContraints::new(evaluated_plonk.node_node_equivalences);
        interpret_plonk_node_to_execution_trace_table_helper(
            &evaluated_plonk.root,
            &mut constraints,
        );

        let wire_equivalences = build_permutation_groups(&mut constraints);
        let permutation_map =
            build_permutation_map(&wire_equivalences, &constraints.gate_operations);

        // Verify all cells in each equivalence group have the same value
        fn check_wire_equivalence_groups<F: Field>(
            wire_equivalences: &[HashSet<PositionCell>],
            gate_operations: &[ExecutionRow<F, 3>],
        ) {
            for (group_id, cells) in wire_equivalences.iter().enumerate() {
                let values: Vec<F> = cells
                    .iter()
                    .map(|cell| {
                        // Match wire type to get correct cell value from execution trace
                        return match cell.wire_type {
                            ColumnType::Input(0) => gate_operations[cell.row_idx].inputs[0].value,
                            ColumnType::Input(1) => gate_operations[cell.row_idx].inputs[1].value,
                            ColumnType::Input(2) => gate_operations[cell.row_idx].inputs[2].value,
                            ColumnType::Output => gate_operations[cell.row_idx].output.value,
                            _ => panic!("Invalid wire type"),
                        };
                    })
                    .collect();

                assert!(
                    values.windows(2).all(|w| w[0] == w[1]),
                    "Cells in group {} have inconsistent values: {:?}",
                    group_id,
                    values
                );
            }
        }

        // Convert HashMap values to Vec<HashSet> for ordered processing
        let groups: Vec<HashSet<PositionCell>> = wire_equivalences.values().cloned().collect();
        check_wire_equivalence_groups(&groups, &constraints.gate_operations);
        // Bijection validation
        let mut value_counts = HashMap::new();
        for v in permutation_map.values() {
            *value_counts.entry(v).or_insert(0) += 1;
        }

        for (val, count) in value_counts {
            assert_eq!(
                count, 1,
                "Value {:?} appears {} times for input '{}'",
                val, count, input
            );
        }

        let keys: HashSet<_> = permutation_map.keys().collect();
        let values: HashSet<_> = permutation_map.values().collect();
        assert_eq!(keys, values, "Keys/values mismatch for input '{}'", input);
    }

    #[test]
    fn permutation_bijection_simple_arithmetic() {
        run_permutation_bijection_test("1 + 2 * 3");
        run_permutation_bijection_test("(1 + 2) * 3");
        run_permutation_bijection_test("1 + 2 + 3");
    }

    #[test]
    fn permutation_bijection_variable_reuse() {
        run_permutation_bijection_test("let x = 5 in x + x");
        run_permutation_bijection_test("let x = 5 in let y = x in x + y");
        run_permutation_bijection_test("let x = 2 in x + x + x");
    }

    #[test]
    fn permutation_bijection_boolean_ops() {
        run_permutation_bijection_test("true && (false || true)");
        run_permutation_bijection_test("let x = true in x || false");
    }

    #[test]
    fn permutation_bijection_complex_examples() {
        run_permutation_bijection_test(
            "let compose = fun f -> fun g -> fun x -> f (g x) in \
             let add1 = fun x -> x + 1 in \
             let mul2 = fun x -> x * 2 in \
             compose add1 mul2 5",
        );
    }

    #[test]
    fn permutation_bijection_edge_cases() {
        run_permutation_bijection_test("5"); // Single constant
        run_permutation_bijection_test("let x = 5 in 3"); // Unused binding
        run_permutation_bijection_test("true"); // Boolean literal
    }

    #[test]
    fn basic_single_row_test() {
        let node_0 = int(0, 0);
        let node_1 = int(1, 1);
        let node_2 = add(node_0, node_1, 2);

        let evaluated_plonk = EvaluatedPlonk {
            root: node_2,
            node_node_equivalences: Vec::new(),
        };

        let mut plonk_constraints =
            PlonkContraints::<F>::new(evaluated_plonk.node_node_equivalences);
        interpret_plonk_node_to_execution_trace_table_helper::<F>(
            &evaluated_plonk.root,
            &mut plonk_constraints,
        );

        // Get wire cell reference counts for all nodes in the circuit
        let node_ids = vec![0, 1, 2]; // Node IDs from our test setup
        let wire_counts = count_wire_cell_references(&mut plonk_constraints, &node_ids);

        // Create expected ground truth map
        let expected_counts = HashMap::from([
            (0, 3), // Node 0: 2 references (int cell + add input)
            (1, 3), // Node 1: 2 references (int cell + add input)
            (2, 1), // Node 2: 1 reference (output cell)
        ]);

        let mut wire_counts_vec: Vec<_> = wire_counts.iter().collect();
        wire_counts_vec.sort();
        let mut expected_counts_vec: Vec<_> = expected_counts.iter().collect();
        expected_counts_vec.sort();

        assert_eq!(
            wire_counts_vec, expected_counts_vec,
            "Wire cell reference counts do not match expected values"
        );
    }

    #[test]
    fn multi_operation_test() {
        // Create AST nodes following mathematical precedence
        // 1 + (2 * 8) + 7
        let node_1 = int(1, 0); // Node 0: 1
        let node_2 = int(2, 1); // Node 1: 2
        let node_8 = int(8, 2); // Node 2: 8
        let node_mul = mult(node_2, node_8, 3); // Node 3: 2*8
        let node_add1 = add(node_1, node_mul, 4); // Node 4: 1 + (2*8)
        let node_7 = int(7, 5); // Node 5: 7
        let root_node = add(node_add1, node_7, 6); // Node 6: (1+16) +7

        let evaluated_plonk = EvaluatedPlonk {
            root: root_node,
            node_node_equivalences: Vec::new(),
        };

        let mut plonk_constraints =
            PlonkContraints::<F>::new(evaluated_plonk.node_node_equivalences);
        interpret_plonk_node_to_execution_trace_table_helper::<F>(
            &evaluated_plonk.root,
            &mut plonk_constraints,
        );

        // Verify all nodes are processed
        let node_ids = vec![0, 1, 2, 3, 4, 5, 6];
        let wire_counts = count_wire_cell_references(&mut plonk_constraints, &node_ids);

        // Expected reference counts:
        // - Input nodes (0,1,2,5) appear once each as inputs
        // - Intermediate results (3,4) appear twice (as output and input)
        // - Final result (6) appears once
        let expected_counts =
            HashMap::from([(0, 3), (1, 3), (2, 3), (3, 2), (4, 2), (5, 3), (6, 1)]);

        let mut wire_counts_vec: Vec<_> = wire_counts.iter().collect();
        wire_counts_vec.sort();
        let mut expected_counts_vec: Vec<_> = expected_counts.iter().collect();
        expected_counts_vec.sort();

        assert_eq!(
            wire_counts_vec, expected_counts_vec,
            "Wire cell reference counts do not match expected values"
        );
    }

    #[test]
    fn variable_reuse_test() {
        // Create nodes for: let x = 2 in x + x + x
        let node_2 = int(2, 0); // Node 0: 2 (bound value)
        let var_x1 = var("x", 1, F::from(2u64)); // Node 1: x
        let var_x2 = var("x", 2, F::from(2u64)); // Node 2: x
        let var_x3 = var("x", 3, F::from(2u64)); // Node 3: x
        let add1 = add(var_x1, var_x2, 4); // Node 4: x + x
        let add2 = add(add1, var_x3, 5); // Node 5: (x + x) + x
        let let_node = let_("x", node_2, add2, 6); // Node 6: let binding

        let evaluated_plonk = EvaluatedPlonk {
            root: let_node,
            node_node_equivalences: vec![
                // All x references point to the bound value's node (0)
                (1, 0),
                (2, 0),
                (3, 0),
                (6, 5),
            ],
        };

        let mut plonk_constraints =
            PlonkContraints::<F>::new(evaluated_plonk.node_node_equivalences);
        interpret_plonk_node_to_execution_trace_table_helper::<F>(
            &evaluated_plonk.root,
            &mut plonk_constraints,
        );

        let node_ids = vec![0, 1, 2, 3, 4, 5, 6];
        let wire_counts = count_wire_cell_references(&mut plonk_constraints, &node_ids);

        // Expected reference pattern:
        // - Bound value (node 0) should have 4 references (initial definition + 3 uses)
        // - Variable nodes (1,2,3) share equivalence with node 0
        // - Intermediate adds (4,5) should have standard operation counts
        let expected_counts = HashMap::from([
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5), // All x references grouped
            (4, 2),
            (5, 1),
            (6, 1), // The let node and the body of the let node should be equivalent and reference the same group of wires.
        ]);

        // Convert to vectors and sort them before comparing
        let mut wire_counts_vec: Vec<_> = wire_counts.iter().collect();
        wire_counts_vec.sort();
        let mut expected_counts_vec: Vec<_> = expected_counts.iter().collect();
        expected_counts_vec.sort();

        assert_eq!(
            wire_counts_vec, expected_counts_vec,
            "Variable reuse pattern not properly constrained"
        );
    }

    #[test]
    fn nested_let_reuse_test() {
        // Create nodes for: let x = 2 in let y = 3 in let z = x+y in z + x + y
        let node_2 = int(2, 0); // Node 0: x = 2
        let node_3 = int(3, 1); // Node 1: y = 3
        let var_x = var("x", 2, F::from(2u64)); // Node 2: x
        let var_y1 = var("y", 3, F::from(3u64)); // Node 3: y
        let add_xy = add(var_x, var_y1, 4); // Node 4: x+y
        let let_z = let_(
            "z",
            add_xy,
            {
                let var_z = var("z", 5, F::from(5u64)); // Node 5: z
                let var_x2 = var("x", 6, F::from(2u64)); // Node 6: x
                let var_y2 = var("y", 7, F::from(3u64)); // Node 7: y
                let add_zx = add(var_z, var_x2, 8); // Node 8: z+x
                add(add_zx, var_y2, 9) // Node 9: (z+x)+y
            },
            10,
        );
        let let_y = let_("y", node_3, let_z, 11);
        let let_x = let_("x", node_2, let_y, 12);

        let evaluated_plonk = EvaluatedPlonk {
            root: let_x,
            node_node_equivalences: vec![
                // x references (2,6) -> bound value (0)
                (2, 0),
                (6, 0),
                // y references (3,7) -> bound value (1)
                (3, 1),
                (7, 1),
                // z reference (5) -> bound value (4)
                (5, 4),
                // Let nodes equivalent to their bodies
                (10, 9),
                (11, 10),
                (12, 11),
            ],
        };

        let mut plonk_constraints =
            PlonkContraints::<F>::new(evaluated_plonk.node_node_equivalences);
        interpret_plonk_node_to_execution_trace_table_helper::<F>(
            &evaluated_plonk.root,
            &mut plonk_constraints,
        );

        let node_ids = (0..=12).collect::<Vec<_>>();
        let wire_counts = count_wire_cell_references(&mut plonk_constraints, &node_ids);

        // Expected reference pattern:
        // - x (0): 1 def + 2 uses = 4
        // - y (1): 1 def + 2 uses = 4
        // - z (4): 1 def + 1 uses = 2
        // - Operations: 4,8,9 have standard op counts
        // - Let nodes: 10,11,12 have single refs
        let expected_counts = HashMap::from([
            (0, 4),
            (2, 4),
            (6, 4), // x references
            (1, 4),
            (3, 4),
            (7, 4), // y references
            (4, 2),
            (5, 2), // z references
            (8, 2),
            (9, 1), // Intermediate adds
            (10, 1),
            (11, 1),
            (12, 1), // Let nodes
        ]);

        // Convert to sorted vectors for comparison
        let mut counts: Vec<_> = wire_counts.iter().collect();
        counts.sort();
        let mut expected: Vec<_> = expected_counts.iter().collect();
        expected.sort();

        assert_eq!(
            counts, expected,
            "Nested let references not properly constrained"
        );
    }

    #[test]
    fn let_with_unused_binding_test() {
        // Create nodes for: let x = 2 in 2 + 5
        let node_2 = int(2, 0); // Node 0: 2 (bound value)
        let add_node = add(int(3, 1), int(5, 2), 3); // Node 2: 2 + 5
        let let_node = let_("x", node_2.clone(), add_node.clone(), 4); // Node 3: let binding

        let evaluated_plonk = EvaluatedPlonk {
            root: let_node.clone(),
            node_node_equivalences: vec![(3, 4)],
        };

        let mut plonk_constraints =
            PlonkContraints::<F>::new(evaluated_plonk.node_node_equivalences);
        interpret_plonk_node_to_execution_trace_table_helper::<F>(
            &evaluated_plonk.root,
            &mut plonk_constraints,
        );

        let node_ids = vec![0, 1, 2, 3, 4];
        let wire_counts = count_wire_cell_references(&mut plonk_constraints, &node_ids);

        // Expected reference pattern:
        // - node_0 (2): 2 references (int cell + add input)
        // - node_1 (5): 2 references (int cell + add input)
        // - node_2 (add): 1 reference (output cell)
        // - node_3 (let): shares output cell with node_2
        let expected_counts = HashMap::from([
            (0, 2), // 2 appears in its own int row and as add input
            (1, 3), // 2 appears in its own int row and as add input
            (2, 3), // 5 appears in its own int row and as add input
            (3, 1), // let node shares add's output cell
            (4, 1), // let node shares add's output cell
        ]);

        // Convert to vectors and sort for comparison
        let mut wire_counts_vec: Vec<_> = wire_counts.iter().collect();
        wire_counts_vec.sort();
        let mut expected_counts_vec: Vec<_> = expected_counts.iter().collect();
        expected_counts_vec.sort();

        assert_eq!(
            wire_counts_vec, expected_counts_vec,
            "Unused let binding references not properly constrained"
        );
    }
}
