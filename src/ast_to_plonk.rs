use std::collections::HashMap;
use crate::language_frontend::ast::{Expr, BinOp, Value};
use crate::plonk_circuit::{PlonkNode, NodeMeta, EvalValue};

#[derive(Debug)]
pub enum ConversionError {
    FunctionInFinalResult,
    UnboundVariable(String),
    TypeError(&'static str),
}

/// Environment for converting AST to PlonkNode
#[derive(Debug, Clone)]
struct ConversionEnv {
    /// Maps variable names to their definitions (for inlining)
    definitions: HashMap<String, Expr>,
    /// Counter for generating fresh variable names during inlining
    fresh_var_counter: usize,
}

impl ConversionEnv {
    fn new() -> Self {
        ConversionEnv {
            definitions: HashMap::new(),
            fresh_var_counter: 0,
        }
    }

    fn fresh_var(&mut self) -> String {
        let var = format!("_v{}", self.fresh_var_counter);
        self.fresh_var_counter += 1;
        var
    }
}

/// Convert an AST expression to a PlonkNode, inlining function definitions
pub fn convert_to_plonk(expr: &Expr) -> Result<PlonkNode<()>, ConversionError> {
    let mut env = ConversionEnv::new();
    convert_expr(expr, &mut env)
}

fn convert_expr(expr: &Expr, env: &mut ConversionEnv) -> Result<PlonkNode<()>, ConversionError> {
    match expr {
        Expr::Int(n) => Ok(PlonkNode::Int(*n, ())),
        
        Expr::Bool(b) => Ok(PlonkNode::Bool(*b, ())),
        
        Expr::Var(name) => {
            // If variable has a definition, inline it
            if let Some(def) = env.definitions.get(name).cloned() {
                convert_expr(&def, env)
            } else {
                Err(ConversionError::UnboundVariable(name.clone()))
            }
        }
        
        Expr::BinOp(op, e1, e2) => {
            let p1 = convert_expr(e1, env)?;
            let p2 = convert_expr(e2, env)?;
            
            match op {
                BinOp::Add => Ok(PlonkNode::Add(Box::new(p1), Box::new(p2), ())),
                BinOp::Sub => Ok(PlonkNode::Sub(Box::new(p1), Box::new(p2), ())),
                BinOp::Mul => Ok(PlonkNode::Mult(Box::new(p1), Box::new(p2), ())),
                BinOp::And => {
                    // a && b = a == b
                    Ok(PlonkNode::Eq(Box::new(p1), Box::new(p2), ()))
                }
                BinOp::Or => {
                    // a || b = !(!(a) == !(b))
                    let not_p1 = PlonkNode::Not(Box::new(p1), ());
                    let not_p2 = PlonkNode::Not(Box::new(p2), ());
                    Ok(PlonkNode::Not(
                        Box::new(PlonkNode::Eq(Box::new(not_p1), Box::new(not_p2), ())),
                        (),
                    ))
                }
            }
        }
        
        Expr::Let(name, e1, e2) => {
            // Convert e1 first
            let converted_e1 = convert_expr(e1, env)?;
            
            // Save old binding if it exists
            let old_def = env.definitions.remove(name);
            
            // Add new binding
            env.definitions.insert(name.clone(), (**e1).clone());
            
            // Convert body with new binding
            let result = convert_expr(e2, env);
            
            // Restore old binding or remove
            if let Some(old) = old_def {
                env.definitions.insert(name.clone(), old);
            } else {
                env.definitions.remove(name);
            }
            
            result
        }
        
        Expr::Lam(_, _) => Err(ConversionError::FunctionInFinalResult),
        
        Expr::App(func, arg) => {
            // First convert the function expression
            match &**func {
                Expr::Lam(param, body) => {
                    // Convert the argument first
                    let converted_arg = convert_expr(arg, env)?;
                    
                    // Save old binding if it exists
                    let old_def = env.definitions.remove(param);
                    
                    // Add argument binding using the converted value
                    env.definitions.insert(param.clone(), Expr::Int(match converted_arg {
                        PlonkNode::Int(n, _) => n,
                        _ => return Err(ConversionError::TypeError("Expected integer argument")),
                    }));
                    
                    // Convert body with argument binding
                    let result = convert_expr(body, env);
                    
                    // Restore old binding or remove
                    if let Some(old) = old_def {
                        env.definitions.insert(param.clone(), old);
                    } else {
                        env.definitions.remove(param);
                    }
                    
                    result
                }
                _ => {
                    // If not a direct lambda, evaluate function expression first
                    let func_converted = convert_expr(func, env)?;
                    match func_converted {
                        PlonkNode::If(cond, then_branch, else_branch, _) => {
                            // Handle if-then-else specially
                            Ok(PlonkNode::If(
                                cond,
                                then_branch,  // Already a Box<PlonkNode>
                                else_branch,  // Already a Box<PlonkNode>
                                (),
                            ))
                        }
                        _ => Err(ConversionError::TypeError("Expected a function")),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_frontend::{lexer::lex, parser};
    use crate::language_frontend::evaluator::eval;
    use crate::language_frontend::ast::Value;
    use crate::plonk_circuit::{eval_plonk_node, EvalValue};

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
    unsafe fn test_conversion(input: &str) {
        let ast = parse_expr(input);

        // Interpret with the AST-based evaluator
        let interpreter_result = eval(&ast)
            .unwrap_or_else(|e| panic!("Interpreter error: {:?}", e));

        // Convert to a PlonkNode
        let plonk_node = convert_to_plonk(&ast)
            .unwrap_or_else(|e| panic!("Conversion error: {:?}", e));

        // Evaluate the PlonkNode
        let evaluated_plonk = eval_plonk_node(&plonk_node)
            .unwrap_or_else(|e| panic!("PlonkNode evaluation error: {:?}", e));

        // Extract the evaluated value
        let plonk_value = evaluated_plonk.meta().evaluated_value.clone();

        // Compare results and panic on mismatch
        match interpreter_result {
            Value::Int(i1) => match plonk_value {
                EvalValue::IntVal(i2) => {
                    if i1 != i2 {
                        panic!("Value mismatch: interpreter={}, plonk={}", i1, i2);
                    }
                }
                _ => panic!("Expected integer EvalValue"),
            },
            Value::Bool(b1) => match plonk_value {
                EvalValue::BoolVal(b2) => {
                    if b1 != b2 {
                        panic!("Value mismatch: interpreter={}, plonk={}", b1, b2);
                    }
                }
                _ => panic!("Expected boolean EvalValue"),
            },
            Value::Closure(_, _, _) => {
                panic!("Unexpected closure in final result");
            }
        }
    }

    #[test]
    fn test_simple_arithmetic() {
        unsafe { test_conversion("1 + 2 * 3"); }
        unsafe { test_conversion("(1 + 2) * 3"); }
        unsafe { test_conversion("1 + 2 + 3"); }
    }

    #[test]
    fn test_let_bindings() {
        unsafe { test_conversion("let x = 5 in x + 3"); }
        unsafe { test_conversion("let x = 5 in let y = 3 in x + y"); }
    }

    #[test]
    fn test_function_application() {
        unsafe { test_conversion("let f = fun x -> x + 1 in f 5"); }
        unsafe { test_conversion("let add = fun x -> fun y -> x + y in (add 3) 4"); }
    }

    #[test]
    fn test_boolean_operations() {
        unsafe { test_conversion("true && false"); }
        unsafe { test_conversion("true || false"); }
        unsafe { test_conversion("let x = true in let y = false in x && y"); }
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
        unsafe { test_conversion(input); }
    }

    #[test]
    fn test_function_errors() {
        // Ensure we directly see the underlying error for a final function
        let expr = parse_expr("fun x -> x + 1");
        let result = convert_to_plonk(&expr);
        assert!(result.is_err(), "Expected error but got Ok");
    }
} 