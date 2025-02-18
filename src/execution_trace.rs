use std::collections::{HashMap, HashSet};

use ark_ff::{FftField, Field, One, Zero};
use ark_poly::univariate::DensePolynomial;
use ark_poly::EvaluationDomain;
use ark_poly::{GeneralEvaluationDomain, Polynomial};

use crate::language_frontend::Expr;
use crate::plonk_circuit::{EvalValue, NodeMeta, PlonkNode, PlonkNodeKind};
use crate::offset_table::OffsetTable;
use crate::polynomial_utils::evaluations_to_dense_polynomial;

// Each expression in the AST will be tied to an ID
type PlonkNodeId = usize;

struct ExecutionCell<F> {
    node_id: PlonkNodeId, // It would be helpful to know where the node came from
    value: F
}

pub struct ExecutionRow<F, const N: usize> {
    pub operation: PlonkNodeKind,
    pub inputs: [F; N], // fixed-size array of inputs
    pub output: F,
}

pub struct ExecutionTable<F, const N: usize> {
    pub rows: Vec<ExecutionRow<F, N>>,
}

pub struct PlonkContraints<F> {
    pub gate_operationss: Vec<ExecutionRow<F, 3>>,
    pub node_equivalences: Vec<(PlonkNodeId, PositionCell)>, // This tells us if a plonk node expression is equivalent to a wire 
}

impl<F> PlonkContraints<F> {
    pub fn new() -> Self {
        Self {
            gate_operationss: Vec::new(),
            node_equivalences: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ColumnType {
    Input(i32),
    Output,
}

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

// This contains the evaluation for evaluating the permutation polynomials
struct PlonkEnv {
    /// Maps variable names to their definitions (for inlining)
    definitions: HashMap<String, Expr>,
    /// Counter for generating fresh variable names during inlining
    fresh_var_counter: usize,
}


/// This helper function is intended to interpret a `PlonkNode<NodeMeta<EvalValue>>`
/// and adds the constraints of each node into a table.
/// It will likely gather information about inputs, outputs, and any intermediate
/// nodes, then populate rows in the execution trace table. Currently unimplemented.
fn interpret_plonk_node_to_execution_trace_table_helper<F: Field>(
    node: &PlonkNode<NodeMeta<F>>,
    constraints: &PlonkContraints<F>,
) -> () {
    // Placeholder implementation: this function still needs its logic to
    // generate and populate an ExecutionTraceTable with the necessary information.
    // Future steps might include handling each node type, capturing relevant field elements,
    // and mapping them to their respective row positions.
    todo!()
}


pub fn interpret_plonk_node_to_execution_trace_table<F: Field>(node: &PlonkNode<NodeMeta<F>>) -> ExecutionTraceTable<F> {
    todo!()
}

// First you build the grouping of the wire positions by node id by building the wire cell equivalences.
// From these groupings, you then build the permutations together
fn build_permutation_map<F: Field>(trace: &PlonkContraints<F>) -> PermutationMap {
    todo!()
}


// You would need to first build the permutation map, then you should be able to build the execution trace table
fn build_execution_trace_table<F: Field>(trace: &PlonkContraints<F>) -> ExecutionTraceTable<F> {
    todo!()
}


/// Example: Implementing Display for ExecutionTraceTable so we can prettily
/// print out PLONK constraints (or trace rows) as a string.
///
/// Suppose ExecutionTraceTable<F> contains a list of rows (or constraints),
/// each representing part of the circuit's execution trace. Then we can
/// implement Display as follows:
/// Start of Selection
impl<F: Field + std::fmt::Debug> std::fmt::Display for PlonkContraints<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    // We'll assume PlonkContraints<F> has a field called `gates` that holds the gate operations
    // and we want to pretty print them. If your actual field is named differently, adjust accordingly.
    // For demonstration:
    writeln!(f, "PLONK Constraints:")?;
    for gate in self.gate_operationss.iter() {
        // Pretty print in the format: operation, [input1, input2, input3], output
        writeln!(
            f,
            "{:?}, [{:?}, {:?}, {:?}], {:?}",
            gate.operation,
            gate.inputs[0],
            gate.inputs[1],
            gate.inputs[2],
            gate.output
        )?;
    }
    Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use ark_ff::Zero; // or FftField, One, etc. if you need them
    use crate::execution_trace::{
        interpret_plonk_node_to_execution_trace_table_helper,
        PlonkContraints,
        PlonkNodeId,
        PositionCell,
        ColumnType,
        PermutationMap,
        WireCellEquivalences,
        build_permutation_map,
        // If you need more items from your crate, add them here
    };
    use crate::plonk_circuit::{eval_plonk_node, EvalValue, NodeMeta, PlonkNode, PlonkNodeKind};
    use ark_ff::Field;
    use crate::language_frontend::{lexer::lex, parser};
    use crate::ast_to_plonk::{convert_to_plonk};
    use crate::language_frontend::ast::Expr;
    
    fn parse_into_plonk_constraints<F: Field>(input: &str) -> PlonkContraints<F> {
        let ast = parse_expr(input);
        let plonk_node = convert_to_plonk(&ast)
            .unwrap_or_else(|e| panic!("Conversion error: {:?}", e));
        let evaluated_plonk = eval_plonk_node::<F>(&plonk_node)
            .unwrap_or_else(|e| panic!("PlonkNode evaluation error: {:?}", e));

        let plonk_constraints = PlonkContraints::<F>::new();

        interpret_plonk_node_to_execution_trace_table_helper::<F>(&evaluated_plonk, &plonk_constraints);
        
        plonk_constraints
    }

    /// Helper function to build a minimal circuit (represented as a `PlonkNode`)
    /// from a user-level program or raw data for testing. This is just a stub.
    /// In your real code, you'll likely parse or directly construct the AST
    /// from the user program, then compile it into a `PlonkNode<NodeMeta<F>>`.
    fn build_test_circuit_single_row<F: Field>() -> PlonkContraints<F> {
        // For illustration, pretend this is: x + y = z
        // Return a single PlonkNode with the operation "Add"
        // and metadata describing x, y, and z.
        // This is just a stub with dummy values. Adapt as needed.
        parse_into_plonk_constraints("1 + 2")
    }


    use ark_bn254::Fr as F;

    #[test]
    fn basic_single_row_test() {
        // 1) Build/compile a minimal circuit with one row, e.g. x + y = z
        // 2) Convert into Plonk constraints
        // 3) Call interpret_plonk_node_to_execution_trace_table_helper (or the table-building logic)
        // 4) Check constraints + permutation map

        let plonk_constraints = parse_into_plonk_constraints::<F>("1 + 2");
        println!("{plonk_constraints}");

        // let constraints = PlonkContraints {
        //     gate_operationss: vec![],
        //     node_equivalences: vec![],
        // };

        // interpret_plonk_node_to_execution_trace_table_helper(&root_node, &constraints);
        // Then build the actual trace table or get it from your function

        // For demonstration, we just assert no panic:
        assert!(true);
    }

    #[test]
    fn small_multi_row_test() {
        let plonk_constraints = parse_into_plonk_constraints::<F>("1 + 2 * 8 + 7");
        println!("{plonk_constraints}");
        // let root_node = build_test_circuit_multi_row::<ark_ff::Fp256<ark_ff::MontBackend>>();
        // let constraints = PlonkContraints {
        //     gate_operationss: vec![],
        //     node_equivalences: vec![],
        // };

        // interpret_plonk_node_to_execution_trace_table_helper(&root_node, &constraints);

        assert!(true);
    }

    #[test]
    fn wire_equivalence_test() {

        let plonk_constraints = parse_into_plonk_constraints::<F>("let x = 2 in x + x + x");
        println!("{plonk_constraints}");

        // let root_node = build_test_circuit_wire_equivalence::<ark_ff::Fp256<ark_ff::MontBackend>>();
        // let constraints = PlonkContraints {
        //     gate_operationss: vec![],
        //     node_equivalences: vec![],
        // };

        // // interpret_plonk_node_to_execution_trace_table_helper(&root_node, &constraints);

        // let _permutation_map = build_permutation_map(&constraints);
        // Check the repeated variable -> cycle
        assert!(true);
    }

    #[test]
    fn simple_use_variable_multiple_times_which_leads_to_cycle_more_than_two_test() {

        let plonk_constraints = parse_into_plonk_constraints::<F>("let x = 2 in let y = 3 in let z = x + y in z + x + y");
        println!("{plonk_constraints}");
    }

    #[test]
    fn unconnected_wire_test() {
        let plonk_constraints = parse_into_plonk_constraints::<F>("let x = 2 in 2 + 5");
        println!("{plonk_constraints}");
    }

    fn parse_expr(input: &str) -> Box<Expr> {
        let tokens = lex(input);
        let token_triples: Vec<_> = tokens.into_iter().enumerate()
            .map(|(i, t)| (i, t, i + 1))
            .collect();
        
        parser::ExprParser::new()
            .parse(token_triples.into_iter())
            .unwrap()
    }

}

