use std::collections::HashMap;
use std::marker::PhantomData;

/// Metadata for each node: an ID and a definite evaluated value (no Option).
#[derive(Debug, Clone)]
pub struct NodeMeta {
    pub id: usize,
    pub evaluated_value: EvalValue,
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

/// Simple struct for environment (assign IDs, hold cache if you like).
/// We'll keep a cache that maps the pointer of the “old” AST<()> node
/// to the newly transformed AST<NodeMeta>.
struct Env {
    next_id: usize,
    cache: HashMap<*const AST<()>, AST<NodeMeta>>,
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
impl AST<NodeMeta> {
    pub fn meta(&self) -> &NodeMeta {
        match self {
            AST::Int(_, meta)
            | AST::Bool(_, meta)
            | AST::Add(_, _, meta)
            | AST::Sub(_, _, meta)
            | AST::Mult(_, _, meta)
            | AST::Div(_, _, meta)
            | AST::Eq(_, _, meta)
            | AST::Not(_, meta)
            | AST::If(_, _, _, meta) => meta,
        }
    }
}

/// Our main transformation: AST<()> -> Result<AST<NodeMeta>, EvalError>
/// Now uses a cache to avoid recomputing the same AST node multiple times.
/// (Most pure AST trees won't share the exact same pointer for subtrees,
/// but if you do have shared subtrees, the cache will skip repeated work.)
fn add_metadata_and_eval(expr: &AST<()>, env: &mut Env) -> Result<AST<NodeMeta>, EvalError> {
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
                    id,
                    evaluated_value: EvalValue::IntVal(*value),
                },
            )
        }
        AST::Bool(b, ()) => {
            let id = env.fresh_id();
            AST::Bool(
                *b,
                NodeMeta {
                    id,
                    evaluated_value: EvalValue::BoolVal(*b),
                },
            )
        }

        // -----------------
        // Arithmetic
        // -----------------
        AST::Add(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval(rhs, env)?;
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
                    id,
                    evaluated_value,
                },
            )
        }

        AST::Sub(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval(rhs, env)?;
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
                    id,
                    evaluated_value,
                },
            )
        }

        AST::Mult(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval(rhs, env)?;
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
                    id,
                    evaluated_value,
                },
            )
        }

        AST::Div(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval(rhs, env)?;
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
                    id,
                    evaluated_value,
                },
            )
        }

        // -----------------
        // Comparisons
        // -----------------
        AST::Eq(lhs, rhs, ()) => {
            let lhs_w_meta = add_metadata_and_eval(lhs, env)?;
            let rhs_w_meta = add_metadata_and_eval(rhs, env)?;
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
                    id,
                    evaluated_value,
                },
            )
        }

        // -----------------
        // Boolean NOT
        // -----------------
        AST::Not(sub, ()) => {
            let sub_w_meta = add_metadata_and_eval(sub, env)?;
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
                    id,
                    evaluated_value,
                },
            )
        }

        // -----------------
        // If-then-else
        // -----------------
        AST::If(cond, then_branch, else_branch, ()) => {
            let cond_w_meta = add_metadata_and_eval(cond, env)?;
            let then_w_meta = add_metadata_and_eval(then_branch, env)?;
            let else_w_meta = add_metadata_and_eval(else_branch, env)?;
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
                    id,
                    evaluated_value,
                },
            )
        }
    };

    // Store the newly computed AST in the cache before returning.
    env.cache.insert(expr_ptr, result_ast.clone());
    Ok(result_ast)
}

/// A top-level function that creates an Env and calls add_metadata_and_eval.
fn eval_ast(expr: &AST<()>) -> Result<AST<NodeMeta>, EvalError> {
    let mut env = Env::new();
    add_metadata_and_eval(expr, &mut env)
}

fn main() {
    // Create a tiny AST<()> with no metadata:
    let naked_ast = AST::Add(
        Box::new(AST::Int(40, ())),
        Box::new(AST::Int(2, ())),
        (),
    );

    // Evaluate and return an AST<NodeMeta> or an evaluation error
    match eval_ast(&naked_ast) {
        Ok(evaluated_ast) => {
            println!("Success! Evaluated AST = {:?}", evaluated_ast);
        }
        Err(err) => {
            println!("Evaluation failed: {:?}", err);
        }
    }
}