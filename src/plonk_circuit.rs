use ark_ff::Field;
use std::collections::HashMap;

// Each expression in the AST will be tied to an ID
pub type PlonkNodeId = usize;

/// Metadata for each node: an ID and an evaluated value.
#[derive(Debug, Clone)]
pub struct NodeMeta<V> {
    pub node_id: PlonkNodeId,
    pub evaluated_value: V,
}

/// An example of a small "value" enum (so you can handle integers, booleans, etc.)
#[derive(Debug, Clone)]
pub enum EvalValue {
    IntVal(i64),
    BoolVal(bool),
}

/// Possible errors when evaluating AST.
#[derive(Debug, Clone)]
pub enum EvalError {
    DivisionByZero,
    TypeError(String),
}

#[derive(Debug, Clone)]
pub enum PlonkNode<M> {
    /// Represents an integer constant
    Int(i64, M),

    /// Represents a boolean constant
    Bool(bool, M),

    /// (expr + expr)
    Add(Box<PlonkNode<M>>, Box<PlonkNode<M>>, M),

    /// (expr - expr)
    Sub(Box<PlonkNode<M>>, Box<PlonkNode<M>>, M),

    /// (expr * expr)
    Mult(Box<PlonkNode<M>>, Box<PlonkNode<M>>, M),

    /// (expr / expr)
    Div(Box<PlonkNode<M>>, Box<PlonkNode<M>>, M),

    /// (expr == expr)
    Eq(Box<PlonkNode<M>>, Box<PlonkNode<M>>, M),

    /// Boolean NOT
    Not(Box<PlonkNode<M>>, M),

    /// If-then-else
    If(Box<PlonkNode<M>>, Box<PlonkNode<M>>, Box<PlonkNode<M>>, M),

    // Let binding
    Let(String, Box<PlonkNode<M>>, Box<PlonkNode<M>>, M),

    // Variable
    Var(String, M),
}

#[derive(Debug, Clone)]
pub struct EvaluatedPlonk<F: Field> {
    pub root: PlonkNode<NodeMeta<F>>,
    pub node_node_equivalences: Vec<(PlonkNodeId, PlonkNodeId)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlonkNodeKind {
    Int,
    Bool,
    Add,
    Sub,
    Mult,
    Div,
    Eq,
    Not,
    If,
}

/// Holds node ID and its field value so variables can point to both.
#[derive(Debug, Clone)]
pub struct DefinitionVal<F: Field> {
    pub node_id: PlonkNodeId,
    pub value: F,
}

/// Environment storing field values for each variable name, along with node equivalences.
pub struct Env<F: Field> {
    pub next_id: PlonkNodeId,
    /// Maps variable names -> (node_id, field value)
    pub definitions: HashMap<String, DefinitionVal<F>>,
    pub equivalent_nodes: Vec<(PlonkNodeId, PlonkNodeId)>,
}

impl<F: Field> Env<F> {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            definitions: HashMap::new(),
            equivalent_nodes: Vec::new(),
        }
    }

    pub fn fresh_id(&mut self) -> PlonkNodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Records that two node IDs are semantically equivalent in the final circuit.
    pub fn add_equivalent_nodes(&mut self, node1_id: PlonkNodeId, node2_id: PlonkNodeId) {
        self.equivalent_nodes.push((node1_id, node2_id));
    }
}

/// Helper so AST<NodeMeta> can return its metadata easily.
impl<M> PlonkNode<M> {
    pub fn meta(&self) -> &M {
        match self {
            PlonkNode::Int(_, m)
            | PlonkNode::Bool(_, m)
            | PlonkNode::Add(_, _, m)
            | PlonkNode::Sub(_, _, m)
            | PlonkNode::Mult(_, _, m)
            | PlonkNode::Div(_, _, m)
            | PlonkNode::Eq(_, _, m)
            | PlonkNode::Not(_, m)
            | PlonkNode::If(_, _, _, m)
            | PlonkNode::Let(_, _, _, m)
            | PlonkNode::Var(_, m) => m,
        }
    }
}

/// Our main transformation: AST<()> -> Result<AST<NodeMeta<F>>, EvalError>
/// that evaluates using the Field trait F rather than EvalValue.
fn add_metadata_and_eval_internal<F: Field>(
    expr: &PlonkNode<()>,
    env: &mut Env<F>,
) -> Result<PlonkNode<NodeMeta<F>>, EvalError> {

    let result_ast = match expr {
        // -----------------
        // Constants
        // -----------------
        PlonkNode::Int(value, ()) => {
            let id = env.fresh_id();
            let fval = F::from(*value as u64);
            PlonkNode::Int(
                *value,
                NodeMeta {
                    node_id: id,
                    evaluated_value: fval,
                },
            )
        }
        PlonkNode::Bool(b, ()) => {
            let id = env.fresh_id();
            let fval = if *b { F::one() } else { F::zero() };
            PlonkNode::Bool(
                *b,
                NodeMeta {
                    node_id: id,
                    evaluated_value: fval,
                },
            )
        }

        // -----------------
        // Arithmetic
        // -----------------
        PlonkNode::Add(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();
            let sum = lhs_w_meta.meta().evaluated_value + rhs_w_meta.meta().evaluated_value;

            PlonkNode::Add(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: sum,
                },
            )
        }
        PlonkNode::Sub(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();
            let diff = lhs_w_meta.meta().evaluated_value - rhs_w_meta.meta().evaluated_value;

            PlonkNode::Sub(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: diff,
                },
            )
        }
        PlonkNode::Mult(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();
            let product = lhs_w_meta.meta().evaluated_value * rhs_w_meta.meta().evaluated_value;

            PlonkNode::Mult(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: product,
                },
            )
        }
        PlonkNode::Div(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();
            let denominator: F = rhs_w_meta.meta().evaluated_value;

            if denominator.is_zero() {
                return Err(EvalError::DivisionByZero);
            }

            // Field division
            let quotient = lhs_w_meta.meta().evaluated_value * denominator.inverse().unwrap();
            PlonkNode::Div(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: quotient,
                },
            )
        }

        // -----------------
        // Comparisons
        // -----------------
        PlonkNode::Eq(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();

            // Store result as 1 if equal, 0 otherwise.
            let eq_f = if lhs_w_meta.meta().evaluated_value == rhs_w_meta.meta().evaluated_value {
                F::one()
            } else {
                F::zero()
            };

            PlonkNode::Eq(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: eq_f,
                },
            )
        }

        // -----------------
        // Boolean NOT (interpreted as 0 => true, otherwise => false)
        // We flip 0 <-> non-zero in the field sense.
        // If val == 0 => 1 else => 0
        // This retains the spirit of boolean inversion, but in F.
        // If the input wasn't 0 or 1, it toggles to 0 anyway.
        // We are ignoring integer type checks from old code.
        // This satisfies "use the trait, Field F" approach.
        // 
        // If you want strict boolean checks, you'd have to enforce val == 0 or 1.
        PlonkNode::Not(sub, ()) => {
            let sub_w_meta = add_metadata_and_eval_internal(sub, env)?;
            let id = env.fresh_id();
            let sub_val: F = sub_w_meta.meta().evaluated_value;

            let inverted = if sub_val.is_zero() {
                F::one()
            } else {
                F::zero()
            };

            PlonkNode::Not(
                Box::new(sub_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: inverted,
                },
            )
        }

        // -----------------
        // If-then-else (cond != 0 => then, else => else)
        // for Field-based booleans.
        // 
        PlonkNode::If(cond, then_branch, else_branch, ()) => {
            let cond_w_meta = add_metadata_and_eval_internal::<F>(cond, env)?;
            let then_w_meta = add_metadata_and_eval_internal::<F>(then_branch, env)?;
            let else_w_meta = add_metadata_and_eval_internal::<F>(else_branch, env)?;
            let id = env.fresh_id();

            let chosen = if cond_w_meta.meta().evaluated_value.is_zero() {
                else_w_meta.meta().evaluated_value
            } else {
                then_w_meta.meta().evaluated_value
            };

            PlonkNode::If(
                Box::new(cond_w_meta),
                Box::new(then_w_meta),
                Box::new(else_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: chosen,
                },
            )
        }

        // -----------------
        // Let binding
        // Inlines the definition into env. We store the field result into env
        // by converting it to an EvalValue. Then evaluate the body, restore env.
        // 
        PlonkNode::Let(var_name, def_expr, body_expr, ()) => {
            let def_w_meta = add_metadata_and_eval_internal(def_expr, env)?;

            // Store node ID + value in the env
            let old_def = env.definitions.insert(
                var_name.clone(),
                DefinitionVal {
                    node_id: def_w_meta.meta().node_id,
                    value: def_w_meta.meta().evaluated_value,
                },
            );

            // Evaluate the body
            let body_w_meta = add_metadata_and_eval_internal(body_expr, env)?;

            // Restore old definition if it existed
            if let Some(old_value) = old_def {
                env.definitions.insert(var_name.clone(), old_value);
            } else {
                env.definitions.remove(var_name);
            }

            let id = env.fresh_id();

            // The let node is equivalent to the body node
            env.add_equivalent_nodes(id, body_w_meta.meta().node_id);

            let final_val = body_w_meta.meta().evaluated_value;
            PlonkNode::Let(
                var_name.clone(),
                Box::new(def_w_meta),
                Box::new(body_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value: final_val,
                },
            )
        }

        // -----------------
        // Variable reference
        // We look it up in the environment, convert it to F, and use that.
        // 
        PlonkNode::Var(var_name, ()) => {
            let definition_val = env
                .definitions
                .get(var_name)
                .ok_or_else(|| EvalError::TypeError(format!("Unbound variable: {}", var_name)))?;

            let definition_val = definition_val.clone();
            let id = env.fresh_id();

            // The newly created Var node is equivalent to the definition's node
            env.add_equivalent_nodes(id, definition_val.node_id);

            PlonkNode::Var(
                var_name.clone(),
                NodeMeta {
                    node_id: id,
                    evaluated_value: definition_val.value,
                },
            )
        }
    };

    Ok(result_ast)
}

/// A top-level function that creates an Env and calls `add_metadata_and_eval_internal`.
#[allow(dead_code)]
pub fn eval_plonk_node<F: Field>(root: &PlonkNode<()>) -> Result<EvaluatedPlonk<F>, EvalError> {
    let mut env = Env::new();
    let evaluated = add_metadata_and_eval_internal::<F>(root, &mut env)?;
    Ok(EvaluatedPlonk {
        root: evaluated,
        node_node_equivalences: env.equivalent_nodes,
    })
}

/// Convert a single EvalValue (i64 or bool) into a Field element.
fn map_eval_value_to_field<F: Field>(val: &EvalValue) -> F {
    match val {
        EvalValue::IntVal(i) => F::from(*i as u64), // Convert i64 to u64 since Field implements From<u64>
        EvalValue::BoolVal(b) => {
            if *b {
                F::one()
            } else {
                F::zero()
            }
        }
    }
}

/// Recursively convert each node's metadata from EvalValue → F.
fn map_plonk_node_value_to_field<F: Field>(
    plonk_node: &PlonkNode<NodeMeta<EvalValue>>,
) -> PlonkNode<NodeMeta<F>> {
    match plonk_node {
        PlonkNode::Int(x, meta) => PlonkNode::Int(
            *x,
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Bool(b, meta) => PlonkNode::Bool(
            *b,
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Add(lhs, rhs, meta) => PlonkNode::Add(
            Box::new(map_plonk_node_value_to_field(lhs)),
            Box::new(map_plonk_node_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Sub(lhs, rhs, meta) => PlonkNode::Sub(
            Box::new(map_plonk_node_value_to_field(lhs)),
            Box::new(map_plonk_node_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Mult(lhs, rhs, meta) => PlonkNode::Mult(
            Box::new(map_plonk_node_value_to_field(lhs)),
            Box::new(map_plonk_node_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Div(lhs, rhs, meta) => PlonkNode::Div(
            Box::new(map_plonk_node_value_to_field(lhs)),
            Box::new(map_plonk_node_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Eq(lhs, rhs, meta) => PlonkNode::Eq(
            Box::new(map_plonk_node_value_to_field(lhs)),
            Box::new(map_plonk_node_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Not(sub, meta) => PlonkNode::Not(
            Box::new(map_plonk_node_value_to_field(sub)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::If(cond, then_branch, else_branch, meta) => PlonkNode::If(
            Box::new(map_plonk_node_value_to_field(cond)),
            Box::new(map_plonk_node_value_to_field(then_branch)),
            Box::new(map_plonk_node_value_to_field(else_branch)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Let(var_name, def_expr, body_expr, meta) => PlonkNode::Let(
            var_name.clone(),
            Box::new(map_plonk_node_value_to_field(def_expr)),
            Box::new(map_plonk_node_value_to_field(body_expr)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        PlonkNode::Var(var_name, meta) => PlonkNode::Var(
            var_name.clone(),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
    }
}

pub mod evaluated_node_factory {
    use super::{PlonkNode, NodeMeta, Field};

    /// Create an integer constant node with metadata
    pub fn int<F: Field>(value: i64, node_id: usize) -> PlonkNode<NodeMeta<F>> {
        PlonkNode::Int(
            value,
            NodeMeta {
                node_id,
                evaluated_value: F::from(value as u64),
            }
        )
    }

    /// Create a boolean constant node with metadata
    pub fn bool<F: Field>(value: bool, node_id: usize) -> PlonkNode<NodeMeta<F>> {
        let field_value = if value { F::one() } else { F::zero() };
        PlonkNode::Bool(
            value,
            NodeMeta {
                node_id,
                evaluated_value: field_value,
            }
        )
    }

    /// Create an addition operation node with metadata
    pub fn add<F: Field>(
        lhs: PlonkNode<NodeMeta<F>>,
        rhs: PlonkNode<NodeMeta<F>>,
        node_id: usize,
    ) -> PlonkNode<NodeMeta<F>> {
        let sum = lhs.meta().evaluated_value + rhs.meta().evaluated_value;
        PlonkNode::Add(
            Box::new(lhs),
            Box::new(rhs),
            NodeMeta {
                node_id,
                evaluated_value: sum,
            }
        )
    }

    /// Create a subtraction operation node with metadata
    pub fn sub<F: Field>(
        lhs: PlonkNode<NodeMeta<F>>,
        rhs: PlonkNode<NodeMeta<F>>,
        node_id: usize,
    ) -> PlonkNode<NodeMeta<F>> {
        let diff = lhs.meta().evaluated_value - rhs.meta().evaluated_value;
        PlonkNode::Sub(
            Box::new(lhs),
            Box::new(rhs),
            NodeMeta {
                node_id,
                evaluated_value: diff,
            }
        )
    }

    /// Create a multiplication operation node with metadata
    pub fn mult<F: Field>(
        lhs: PlonkNode<NodeMeta<F>>,
        rhs: PlonkNode<NodeMeta<F>>,
        node_id: usize,
    ) -> PlonkNode<NodeMeta<F>> {
        let product = lhs.meta().evaluated_value * rhs.meta().evaluated_value;
        PlonkNode::Mult(
            Box::new(lhs),
            Box::new(rhs),
            NodeMeta {
                node_id,
                evaluated_value: product,
            }
        )
    }

    /// Create a division operation node with metadata
    pub fn div<F: Field>(
        lhs: PlonkNode<NodeMeta<F>>,
        rhs: PlonkNode<NodeMeta<F>>,
        node_id: usize,
    ) -> PlonkNode<NodeMeta<F>> {
        let quotient = lhs.meta().evaluated_value * rhs.meta().evaluated_value.inverse().unwrap();
        PlonkNode::Div(
            Box::new(lhs),
            Box::new(rhs),
            NodeMeta {
                node_id,
                evaluated_value: quotient,
            }
        )
    }

    /// Create an equality comparison node with metadata
    pub fn eq<F: Field>(
        lhs: PlonkNode<NodeMeta<F>>,
        rhs: PlonkNode<NodeMeta<F>>,
        node_id: usize,
    ) -> PlonkNode<NodeMeta<F>> {
        let eq_value = if lhs.meta().evaluated_value == rhs.meta().evaluated_value {
            F::one()
        } else {
            F::zero()
        };
        PlonkNode::Eq(
            Box::new(lhs),
            Box::new(rhs),
            NodeMeta {
                node_id,
                evaluated_value: eq_value,
            }
        )
    }

    /// Create a boolean NOT node with metadata
    pub fn not<F: Field>(
        operand: PlonkNode<NodeMeta<F>>,
        node_id: usize,
    ) -> PlonkNode<NodeMeta<F>> {
        let inverted = if operand.meta().evaluated_value.is_zero() {
            F::one()
        } else {
            F::zero()
        };
        PlonkNode::Not(
            Box::new(operand),
            NodeMeta {
                node_id,
                evaluated_value: inverted,
            }
        )
    }

    /// Create a variable reference node with metadata
    pub fn var<F: Field>(
        name: &str, 
        node_id: usize,
        value: F  // Value should come from environment equivalences
    ) -> PlonkNode<NodeMeta<F>> {
        PlonkNode::Var(
            name.to_string(),
            NodeMeta {
                node_id,
                evaluated_value: value,
            },
        )
    }

    /// Create a let binding node with metadata
    pub fn let_<F: Field>(
        var_name: &str,
        bound_expr: PlonkNode<NodeMeta<F>>,
        body_expr: PlonkNode<NodeMeta<F>>,
        node_id: usize,
    ) -> PlonkNode<NodeMeta<F>> {
        let final_val = body_expr.meta().evaluated_value;
        PlonkNode::Let(
            var_name.to_string(),
            Box::new(bound_expr),
            Box::new(body_expr),
            NodeMeta {
                node_id,
                evaluated_value: final_val,
            },
        )
    }
}

use ark_bn254::Fr as F;

fn main() {
    // Create a tiny AST<()> with no metadata:
    let naked_ast = PlonkNode::Add(
        Box::new(PlonkNode::Int(40, ())),
        Box::new(PlonkNode::Int(2, ())),
        (),
    );

    match eval_plonk_node::<F>(&naked_ast) {
        Ok(evaluated_ast) => {
            println!("Success! Evaluated AST = {:?}", evaluated_ast);
        }
        Err(err) => {
            println!("Evaluation failed: {:?}", err);
        }
    }
}
