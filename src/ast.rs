use ark_ff::Field;
use std::collections::HashMap;

/// Metadata for each node: an ID and a definite evaluated value (no Option).
#[derive(Debug, Clone)]
pub struct NodeMeta<V> {
    pub node_id: usize,
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
pub enum AST<M> {
    /// Represents an integer constant
    Int(i64, M),

    /// Represents a boolean constant
    Bool(bool, M),

    /// (expr + expr)
    Add(Box<AST<M>>, Box<AST<M>>, M),

    /// (expr - expr)
    Sub(Box<AST<M>>, Box<AST<M>>, M),

    /// (expr * expr)
    Mult(Box<AST<M>>, Box<AST<M>>, M),

    /// (expr / expr)
    Div(Box<AST<M>>, Box<AST<M>>, M),

    /// (expr == expr)
    Eq(Box<AST<M>>, Box<AST<M>>, M),

    /// Boolean NOT
    Not(Box<AST<M>>, M),

    /// If-then-else
    If(Box<AST<M>>, Box<AST<M>>, Box<AST<M>>, M),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ASTKind {
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

/// Simple struct for environment (assign IDs, hold cache if you like).
/// We'll keep a cache that maps the pointer of the "old" AST<()> node
/// to the newly transformed AST<NodeMeta>.
struct Env {
    next_id: usize,
    cache: HashMap<*const AST<()>, AST<NodeMeta<EvalValue>>>
}

impl Env {
    fn new() -> Self {
        Env {
            next_id: 0,
            cache: HashMap::new(),
        }
    }

    /// Get a fresh ID for a node
    fn fresh_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Helper so AST<NodeMeta> can return its metadata easily.
impl<M> AST<M> {
    pub fn meta(&self) -> &M {
        match self {
            AST::Int(_, m)
            | AST::Bool(_, m)
            | AST::Add(_, _, m)
            | AST::Sub(_, _, m)
            | AST::Mult(_, _, m)
            | AST::Div(_, _, m)
            | AST::Eq(_, _, m)
            | AST::Not(_, m)
            | AST::If(_, _, _, m) => m,
        }
    }
}

/// Our main transformation: AST<()> -> Result<AST<NodeMeta<EvalValue>>, EvalError>
/// Now uses a cache to avoid recomputing the same AST node multiple times.
/// (Most pure AST trees won't share the exact same pointer for subtrees,
/// but if you do have shared subtrees, the cache will skip repeated work.)
fn add_metadata_and_eval_internal(expr: &AST<()>, env: &mut Env) -> Result<AST<NodeMeta<EvalValue>>, EvalError> {
    // If we've already computed this node, return the cached result.
    let expr_ptr = expr as *const AST<()>;
    if let Some(already_computed) = env.cache.get(&expr_ptr) {
        return Ok(already_computed.clone());
    }

    let result_ast = match expr {
        // -----------------
        // Constants
        // -----------------
        AST::Int(value, ()) => {
            let id = env.fresh_id();
            AST::Int(
                *value,
                NodeMeta {
                    node_id: id,
                    evaluated_value: EvalValue::IntVal(*value),
                },
            )
        }
        AST::Bool(b, ()) => {
            let id = env.fresh_id();
            AST::Bool(
                *b,
                NodeMeta {
                    node_id: id,
                    evaluated_value: EvalValue::BoolVal(*b),
                },
            )
        }

        // -----------------
        // Arithmetic
        // -----------------
        AST::Add(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();

            // Both sides must be IntVal
            let (lval, rval) = match (
                &lhs_w_meta.meta().evaluated_value,
                &rhs_w_meta.meta().evaluated_value,
            ) {
                (EvalValue::IntVal(lv), EvalValue::IntVal(rv)) => (*lv, *rv),
                _ => {
                    return Err(EvalError::TypeError(
                        "Add requires integer operands".to_string(),
                    ))
                }
            };

            let evaluated_value = EvalValue::IntVal(lval + rval);
            AST::Add(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value,
                },
            )
        }

        AST::Sub(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();

            let (lval, rval) = match (
                &lhs_w_meta.meta().evaluated_value,
                &rhs_w_meta.meta().evaluated_value,
            ) {
                (EvalValue::IntVal(lv), EvalValue::IntVal(rv)) => (*lv, *rv),
                _ => {
                    return Err(EvalError::TypeError(
                        "Sub requires integer operands".to_string(),
                    ))
                }
            };

            let evaluated_value = EvalValue::IntVal(lval - rval);
            AST::Sub(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value,
                },
            )
        }

        AST::Mult(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();

            let (lval, rval) = match (
                &lhs_w_meta.meta().evaluated_value,
                &rhs_w_meta.meta().evaluated_value,
            ) {
                (EvalValue::IntVal(lv), EvalValue::IntVal(rv)) => (*lv, *rv),
                _ => {
                    return Err(EvalError::TypeError(
                        "Mult requires integer operands".to_string(),
                    ))
                }
            };

            let evaluated_value = EvalValue::IntVal(lval * rval);
            AST::Mult(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value,
                },
            )
        }

        AST::Div(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();

            let (lval, rval) = match (
                &lhs_w_meta.meta().evaluated_value,
                &rhs_w_meta.meta().evaluated_value,
            ) {
                (EvalValue::IntVal(lv), EvalValue::IntVal(rv)) => (*lv, *rv),
                _ => {
                    return Err(EvalError::TypeError(
                        "Div requires integer operands".to_string(),
                    ))
                }
            };

            if rval == 0 {
                return Err(EvalError::DivisionByZero);
            }

            let evaluated_value = EvalValue::IntVal(lval / rval);
            AST::Div(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value,
                },
            )
        }

        // -----------------
        // Comparisons
        // -----------------
        AST::Eq(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval_internal(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval_internal(rhs, env)?;
            let id = env.fresh_id();

            let evaluated_value = match (
                &lhs_w_meta.meta().evaluated_value,
                &rhs_w_meta.meta().evaluated_value,
            ) {
                (EvalValue::IntVal(lv), EvalValue::IntVal(rv)) => EvalValue::BoolVal(lv == rv),
                (EvalValue::BoolVal(lb), EvalValue::BoolVal(rb)) => EvalValue::BoolVal(lb == rb),
                _ => {
                    return Err(EvalError::TypeError(
                        "Eq requires both sides be either int or bool".to_string(),
                    ))
                }
            };

            AST::Eq(
                Box::new(lhs_w_meta),
                Box::new(rhs_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value,
                },
            )
        }

        // -----------------
        // Boolean NOT
        // -----------------
        AST::Not(sub, ()) => {
            let sub_w_meta = add_metadata_and_eval_internal(sub, env)?;
            let id = env.fresh_id();

            let sb = match &sub_w_meta.meta().evaluated_value {
                EvalValue::BoolVal(b) => *b,
                _ => {
                    return Err(EvalError::TypeError(
                        "Not requires a bool operand".to_string(),
                    ))
                }
            };

            let evaluated_value = EvalValue::BoolVal(!sb);
            AST::Not(
                Box::new(sub_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value,
                },
            )
        }

        // -----------------
        // If-then-else
        // -----------------
        AST::If(cond, then_branch, else_branch, ()) => {
            let cond_w_meta = add_metadata_and_eval_internal(cond, env)?;
            let then_w_meta = add_metadata_and_eval_internal(then_branch, env)?;
            let else_w_meta = add_metadata_and_eval_internal(else_branch, env)?;
            let id = env.fresh_id();

            // If the condition is a BoolVal, pick one branch for "overall" value
            let evaluated_value = match cond_w_meta.meta().evaluated_value {
                EvalValue::BoolVal(true) => then_w_meta.meta().evaluated_value.clone(),
                EvalValue::BoolVal(false) => else_w_meta.meta().evaluated_value.clone(),
                // Anything else is a type error
                _ => {
                    return Err(EvalError::TypeError(
                        "If condition must be a bool".to_string(),
                    ))
                }
            };

            AST::If(
                Box::new(cond_w_meta),
                Box::new(then_w_meta),
                Box::new(else_w_meta),
                NodeMeta {
                    node_id: id,
                    evaluated_value,
                },
            )
        }
    };

    // Store the newly computed AST in the cache before returning.
    env.cache.insert(expr_ptr, result_ast.clone());
    Ok(result_ast)
}

/// A top-level function that creates an Env and calls add_metadata_and_eval_internal.
fn eval_ast(expr: &AST<()>) -> Result<AST<NodeMeta<EvalValue>>, EvalError> {
    let mut env = Env::new();
    add_metadata_and_eval_internal(expr, &mut env)
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
fn map_ast_value_to_field<F: Field>(ast: &AST<NodeMeta<EvalValue>>) -> AST<NodeMeta<F>> {
    match ast {
        AST::Int(x, meta) => AST::Int(
            *x,
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::Bool(b, meta) => AST::Bool(
            *b,
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::Add(lhs, rhs, meta) => AST::Add(
            Box::new(map_ast_value_to_field(lhs)),
            Box::new(map_ast_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::Sub(lhs, rhs, meta) => AST::Sub(
            Box::new(map_ast_value_to_field(lhs)),
            Box::new(map_ast_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::Mult(lhs, rhs, meta) => AST::Mult(
            Box::new(map_ast_value_to_field(lhs)),
            Box::new(map_ast_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::Div(lhs, rhs, meta) => AST::Div(
            Box::new(map_ast_value_to_field(lhs)),
            Box::new(map_ast_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::Eq(lhs, rhs, meta) => AST::Eq(
            Box::new(map_ast_value_to_field(lhs)),
            Box::new(map_ast_value_to_field(rhs)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::Not(sub, meta) => AST::Not(
            Box::new(map_ast_value_to_field(sub)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
        AST::If(cond, then_branch, else_branch, meta) => AST::If(
            Box::new(map_ast_value_to_field(cond)),
            Box::new(map_ast_value_to_field(then_branch)),
            Box::new(map_ast_value_to_field(else_branch)),
            NodeMeta {
                node_id: meta.node_id,
                evaluated_value: map_eval_value_to_field(&meta.evaluated_value),
            },
        ),
    }
}

/// The public function that:
/// 1) Evaluates the AST (with no metadata) into AST<NodeMeta<EvalValue>>  
/// 2) Maps that to AST<NodeMeta<F>>  
pub fn add_metadata_and_eval<F: Field>(
    expr: &AST<()>,
) -> Result<AST<NodeMeta<F>>, EvalError> {
    // Step 1: Evaluate to AST<NodeMeta<EvalValue>>
    let mut env = Env::new();
    let eval_value_ast = add_metadata_and_eval_internal(expr, &mut env)?;

    // Step 2: Convert EvalValue → F
    let field_ast = map_ast_value_to_field(&eval_value_ast);
    Ok(field_ast)
}

fn main() {
    // Create a tiny AST<()> with no metadata:
    let naked_ast = AST::Add(
        Box::new(AST::Int(40, ())),
        Box::new(AST::Int(2, ())),
        (),
    );

    // Evaluate and return an AST<NodeMeta<EvalValue>> or an evaluation error
    match eval_ast(&naked_ast) {
        Ok(evaluated_ast) => {
            println!("Success! Evaluated AST = {:?}", evaluated_ast);
        }
        Err(err) => {
            println!("Evaluation failed: {:?}", err);
        }
    }
}
