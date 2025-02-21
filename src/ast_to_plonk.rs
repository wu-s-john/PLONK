use std::collections::HashMap;
use crate::language_frontend::{ast::{BinOp, Expr}, type_checker, Type};
// Removed NodeMeta (unused). We keep EvalValue under tests if needed.
#[cfg(test)]
#[allow(unused_imports)]
use crate::plonk_circuit::EvalValue;
use crate::plonk_circuit::PlonkNode;

/// Environment for converting AST to PlonkNode (and for inlining).
///
/// - We store let-bound function definitions, so if we see `Var(x)` that
///   maps to a literal `Lam(...)`, we can inline it and continue reducing.
#[derive(Debug, Clone)]
struct InlineEnv {
    /// Maps variable names to their let-bound expression (especially Lambdas)
    /// to be inlined; e.g. "f" -> Expr::Lam("x", Box::new(...))
    inlined_funcs: HashMap<String, Expr>,
}

impl InlineEnv {
    fn new() -> Self {
        InlineEnv {
            inlined_funcs: HashMap::new()
        }
    }
}

/// Convert an AST expression to a PlonkNode, inlining function definitions.
#[allow(dead_code)]
pub fn convert_to_plonk(expr: &Expr) -> Result<PlonkNode<()>, InlineError> {
    let mut env = InlineEnv::new();
    
    let result = inline_function_expr(expr, &mut env);
    result
}

/// An example error type for inlining errors.
#[derive(Debug)]
pub enum InlineError {
    PartialApplication(String),
}

/// inline_function_expr:
/// Given an Expr, fully inline (beta-reduce) all Lam/App nodes
/// until none remain. The result is a PlonkNode<()> with
/// no Lam or App. This is a simplistic version that assumes
/// no recursion and no partial applications.
fn inline_function_expr(
    expr: &Expr,
    env: &mut InlineEnv
) -> Result<PlonkNode<()>, InlineError> {
    // First, do a repeated beta-reduction on the Expr (removing all Lam/App),
    // also using env to inline let-bound lambdas.
    let reduced_expr = beta_reduce_all(expr, env)?;
    let expr_type = type_checker::type_check(&reduced_expr)
        .map_err(|_| InlineError::PartialApplication("Type check failed".to_string()))?;
    if let Type::Fun(_, _) = expr_type {
        return Err(InlineError::PartialApplication("Expression is a function type".to_string()));
    }

    // Then convert that final, lambda-free expression to PlonkNode<()>.
    Ok(expr_to_plonk(&reduced_expr, env))
}

/// beta_reduce_all:
/// Keep reducing until we reach a normal form (no more `Lam` or `App`),
/// and also inline `Var(x)` if `x` is a let-bound function.
fn beta_reduce_all(expr: &Expr, env: &mut InlineEnv) -> Result<Expr, InlineError> {
    let mut current = expr.clone();
    loop {
        let (new_expr, changed) = beta_reduce_step(&current, env)?;
        if !changed {
            return Ok(new_expr);
        }
        current = new_expr;
    }
}

/// beta_reduce_step:
/// Perform one pass that tries to reduce the top-most Lam/App pairs
/// (standard beta-reduction) and also inline let-bound lambdas.
fn beta_reduce_step(
    expr: &Expr,
    env: &mut InlineEnv
) -> Result<(Expr, bool), InlineError> {
    match expr {
        // Base cases:
        Expr::Int(_) | Expr::Bool(_) => Ok((expr.clone(), false)),

        // If we see `Var(x)` and `x` is a known let-bound lambda in env,
        // replace it. That can enable future (Lam(...) arg) matches.
        Expr::Var(x) => {
            if let Some(inlined_expr) = env.inlined_funcs.get(x) {
                // Inlining the let-bound lambda or expression
                Ok((inlined_expr.clone(), true))
            } else {
                // Not inlined
                Ok((expr.clone(), false))
            }
        }

        // For BinOp, recursively reduce subexprs:
        Expr::BinOp(op, lhs, rhs) => {
            let (lhs_r, lhs_changed) = beta_reduce_step(lhs, env)?;
            let (rhs_r, rhs_changed) = beta_reduce_step(rhs, env)?;
            let changed = lhs_changed || rhs_changed;
            Ok((
                Expr::BinOp(op.clone(), Box::new(lhs_r), Box::new(rhs_r)),
                changed,
            ))
        }

        // For Let, reduce inside bound and body:
        // Then if the bound is a lambda, we can inline it by inserting
        // into env and dropping the let. That way future references to x
        // become the function body.
        Expr::Let(x, bound, body) => {
            let (bound_r, bound_ch) = beta_reduce_step(bound, env)?;
            let (body_r, body_ch) = beta_reduce_step(body, env)?;
            let changed = bound_ch || body_ch;

            // If the bound is a lambda, store it in the env, remove the let:
            if let Expr::Lam(_, _) = bound_r {
                // Insert it into env — future uses of `Var(x)` will be inlined
                env.inlined_funcs.insert(x.clone(), bound_r.clone());
                // Drop the let, so the expression becomes just `body_r`
                Ok((body_r, true))
            } else {
                // Else keep the let
                Ok((
                    Expr::Let(x.clone(), Box::new(bound_r), Box::new(body_r)),
                    changed,
                ))
            }
        }

        // For a Lambda, we just reduce inside the body but do not remove the Lam:
        Expr::Lam(param, lam_body) => {
            let (body_r, body_ch) = beta_reduce_step(lam_body, env)?;
            if body_ch {
                Ok((Expr::Lam(param.clone(), Box::new(body_r)), true))
            } else {
                Ok((expr.clone(), false))
            }
        }

        // The interesting case: (f arg). We check if `f` reduces to a Lam(...).
        Expr::App(f, arg) => {
            // reduce f and arg first
            let (f_r, f_changed) = beta_reduce_step(f, env)?;
            let (arg_r, arg_changed) = beta_reduce_step(arg, env)?;
            let changed_local = f_changed || arg_changed;

            // Now see if f_r is Lam(...) ...
            if let Expr::Lam(param, body) = f_r {
                // We have ((Lam param body) arg_r), so do substitution
                let substituted = substitute(&body, &param, &arg_r, env)?;
                // Because we performed a real reduction, mark changed = true
                Ok((substituted, true))
            } else {
                // If f_r is not a Lam, just reconstruct as (f_r arg_r).
                let new_expr = Expr::App(Box::new(f_r), Box::new(arg_r));
                Ok((new_expr, changed_local))
            }
        }
    }
}

/// A simpler version of capture-avoiding substitution. In practice,
/// you'd do alpha-renaming if "param" exists free in "replacement".
fn substitute(
    body: &Expr,
    param: &str,
    replacement: &Expr,
    env: &mut InlineEnv
) -> Result<Expr, InlineError> {
    match body {
        Expr::Int(_) | Expr::Bool(_) => Ok(body.clone()),

        Expr::Var(x) => {
            if x == param {
                Ok(replacement.clone())
            } else {
                // If x is a different var, check if we can inline anyway?
                // But typically we just keep it as is:
                Ok(body.clone())
            }
        }

        Expr::Lam(lam_param, lam_body) => {
            // If lam_param == param, there's shadowing, so skip in lam_body
            if lam_param == param {
                Ok(body.clone())
            } else {
                let lam_body_sub = substitute(lam_body, param, replacement, env)?;
                Ok(Expr::Lam(lam_param.clone(), Box::new(lam_body_sub)))
            }
        }

        Expr::App(f, arg) => {
            let f_sub = substitute(f, param, replacement, env)?;
            let arg_sub = substitute(arg, param, replacement, env)?;
            Ok(Expr::App(Box::new(f_sub), Box::new(arg_sub)))
        }

        Expr::Let(x, bound, let_body) => {
            if x == param {
                // shadowing again
                let bound_sub = substitute(bound, param, replacement, env)?;
                Ok(Expr::Let(x.clone(), Box::new(bound_sub), let_body.clone()))
            } else {
                let bound_sub = substitute(bound, param, replacement, env)?;
                let body_sub = substitute(let_body, param, replacement, env)?;
                Ok(Expr::Let(x.clone(), Box::new(bound_sub), Box::new(body_sub)))
            }
        }

        Expr::BinOp(op, lhs, rhs) => {
            let lhs_sub = substitute(lhs, param, replacement, env)?;
            let rhs_sub = substitute(rhs, param, replacement, env)?;
            Ok(Expr::BinOp(op.clone(), Box::new(lhs_sub), Box::new(rhs_sub)))
        }
    }
}

/// Once the expression no longer has any Lam or App,
/// we convert it to a PlonkNode<()>.
pub fn expr_to_plonk(expr: &Expr, _env: &InlineEnv) -> PlonkNode<()> {
    match expr {
        Expr::Int(n) => PlonkNode::Int(*n, ()),
        Expr::Bool(b) => PlonkNode::Bool(*b, ()),
        Expr::Var(v) => PlonkNode::Var(v.clone(), ()),

        Expr::BinOp(op, lhs, rhs) => {
            let lhs_plonk = expr_to_plonk(lhs, _env);
            let rhs_plonk = expr_to_plonk(rhs, _env);
            match op {
                BinOp::Add => PlonkNode::Add(Box::new(lhs_plonk), Box::new(rhs_plonk), ()),
                BinOp::Sub => PlonkNode::Sub(Box::new(lhs_plonk), Box::new(rhs_plonk), ()),
                BinOp::Mul => PlonkNode::Mult(Box::new(lhs_plonk), Box::new(rhs_plonk), ()),
                BinOp::And => {
                    // Treat "a && b" as "a == b" for demonstration
                    PlonkNode::Eq(Box::new(lhs_plonk), Box::new(rhs_plonk), ())
                }
                BinOp::Or => {
                    // a || b ≈ !( (!a) == (!b) )
                    let not_lhs = PlonkNode::Not(Box::new(lhs_plonk), ());
                    let not_rhs = PlonkNode::Not(Box::new(rhs_plonk), ());
                    PlonkNode::Not(
                        Box::new(PlonkNode::Eq(Box::new(not_lhs), Box::new(not_rhs), ())),
                        (),
                    )
                }
            }
        }

        Expr::Let(var_name, bound, body) => {
            // We still allow Let in the final expression if it wasn't inlined away.
            let bound_plonk = expr_to_plonk(bound, _env);
            let body_plonk = expr_to_plonk(body, _env);
            PlonkNode::Let(
                var_name.clone(),
                Box::new(bound_plonk),
                Box::new(body_plonk),
                (),
            )
        }

        // If any Lam/App remained, we'd presumably error or ignore,
        // but our beta_reduce_all should remove them all.
        Expr::Lam(_, _) | Expr::App(_, _) => {
            panic!("expr_to_plonk called on an expression that still has Lam/App!")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_frontend::{lexer::lex, parser};
    use crate::language_frontend::evaluator::eval;
    use crate::language_frontend::ast::Value;
    use crate::plonk_circuit::eval_plonk_node;

    fn parse_expr(input: &str) -> Box<Expr> {
        let tokens = lex(input);
        let token_triples: Vec<_> = tokens.into_iter().enumerate()
            .map(|(i, t)| (i, t, i + 1))
            .collect();
        
        parser::ExprParser::new()
            .parse(token_triples.into_iter())
            .unwrap()
    }

    // Now unsafe, and it panics on any error instead of returning a Result
    unsafe fn test_conversion<F: ark_ff::Field>(input: &str) {
        let ast = parse_expr(input);

        // Interpret with the AST-based evaluator
        let interpreter_result = eval(&ast)
            .unwrap_or_else(|e| panic!("Interpreter error: {:?}", e));

        // Convert to a PlonkNode
        let plonk_node = convert_to_plonk(&ast)
            .unwrap_or_else(|e| panic!("Conversion error: {:?}", e));

        // Evaluate the PlonkNode with a generic Field F
        let evaluated_plonk = eval_plonk_node::<F>(&plonk_node)
            .unwrap_or_else(|e| panic!("PlonkNode evaluation error: {:?}", e));

        let evaluate_plonk_node = evaluated_plonk.root;

        // Extract the evaluated value
        let plonk_value = evaluate_plonk_node.meta().evaluated_value.clone();

        // Compare results and panic on mismatch
        // Convert interpreter result to field value for comparison
        let field_value = match interpreter_result {
            Value::Int(n) => F::from(n as u64),
            Value::Bool(b) => if b { F::from(1u64) } else { F::from(0u64) },
            _ => panic!("Unsupported value type for field conversion"),
        };

        assert_eq!(field_value, plonk_value, "PlonkNode evaluation did not match interpreter");
    }

    use ark_bn254::Fr as F;
    
    #[test]
    fn test_simple_arithmetic() {
        unsafe { test_conversion::<F>("1 + 2 * 3"); }
        unsafe { test_conversion::<F>("(1 + 2) * 3"); }
        unsafe { test_conversion::<F>("1 + 2 + 3"); }
    }

    #[test]
    fn test_let_bindings() {
        unsafe { test_conversion::<F>("let x = 5 in x + 3"); }
        unsafe { test_conversion::<F>("let x = 5 in let y = 3 in x + y"); }
    }

    #[test]
    fn test_function_application() {
        unsafe { test_conversion::<F>("let f = fun x -> x + 1 in f 5"); }
        unsafe { test_conversion::<F>("let add = fun x -> fun y -> x + y in (add 3) 4"); }
    }

    #[test]
    fn test_boolean_operations() {
        unsafe { test_conversion::<F>("true && false"); }
        unsafe { test_conversion::<F>("true || false"); }
        unsafe { test_conversion::<F>("let x = true in let y = false in x && y"); }
    }

    #[test]
    fn test_complex_expression() {
        let input = "
            let compose = fun f -> fun g -> fun x -> f (g x) in
            let add1 = fun x -> x + 1 in
            let mul2 = fun x -> x * 2 in
            let composed = compose add1 mul2 in
            composed 5
        ";
        unsafe { test_conversion::<F>(input); }
    }

    #[test]
    fn test_function_errors() {
        // Ensure we directly see the underlying error for a final function
        let expr = parse_expr("fun x -> x + 1");
        let result = convert_to_plonk(&expr);
        assert!(result.is_err(), "Expected error but got Ok");
    }
} 