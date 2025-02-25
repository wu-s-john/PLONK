use std::collections::HashMap;
use crate::language_frontend::ast::{Expr, Value, Environment, BinOp};

#[derive(Debug)]
pub enum EvalError {
    UnboundVariable(String),
    TypeError(&'static str),
}

type EvalResult = Result<Value, EvalError>;

pub fn eval(expr: &Expr) -> EvalResult {
    eval_with_env(expr, &mut HashMap::new())
}

fn eval_with_env(expr: &Expr, env: &mut Environment) -> EvalResult {
    match expr {
        Expr::Int(n) => Ok(Value::Int(*n)),
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        
        Expr::Var(name) => {
            env.get(name)
               .cloned()
               .ok_or_else(|| EvalError::UnboundVariable(name.clone()))
        }
        
        Expr::Lam(param, body) => {
            Ok(Value::Closure(param.clone(), body.clone(), env.clone()))
        }
        
        Expr::App(func, arg) => {
            let func_val = eval_with_env(func, env)?;
            let arg_val = eval_with_env(arg, env)?;
            
            match func_val {
                Value::Closure(param, body, mut closure_env) => {
                    closure_env.insert(param, arg_val);
                    eval_with_env(&body, &mut closure_env)
                }
                _ => Err(EvalError::TypeError("Expected a function")),
            }
        }
        
        Expr::Let(name, expr, body) => {
            let val = eval_with_env(expr, env)?;
            env.insert(name.clone(), val);
            let result = eval_with_env(body, env);
            env.remove(name);
            result
        }

        Expr::If(cond, then_expr, else_expr) => {
            match eval_with_env(cond, env)? {
                Value::Bool(true) => eval_with_env(then_expr, env),
                Value::Bool(false) => eval_with_env(else_expr, env),
                _ => Err(EvalError::TypeError("Condition must be a boolean")),
            }
        }
        
        Expr::BinOp(op, e1, e2) => {
            let v1 = eval_with_env(e1, env)?;
            let v2 = eval_with_env(e2, env)?;
            
            match (op, v1, v2) {
                (BinOp::Add, Value::Int(i), Value::Int(j)) => Ok(Value::Int(i + j)),
                (BinOp::Sub, Value::Int(i), Value::Int(j)) => Ok(Value::Int(i - j)),
                (BinOp::Mul, Value::Int(i), Value::Int(j)) => Ok(Value::Int(i * j)),
                (BinOp::And, Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 && b2)),
                (BinOp::Or, Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 || b2)),
                (BinOp::Eq, Value::Int(i), Value::Int(j)) => Ok(Value::Bool(i == j)),
                (BinOp::Eq, Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 == b2)),
                _ => Err(EvalError::TypeError("Type mismatch in binary operation")),
            }
        }
    }
} 