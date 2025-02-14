use std::collections::HashMap;
use crate::language_frontend::ast::{Expr, Type, BinOp};

#[derive(Debug)]
pub enum TypeError {
    UnboundVariable(String),
    TypeMismatch {
        expected: Type,
        found: Type,
    },
    NotAFunction(Type),
}

type TypeEnv = HashMap<String, Type>;
type TypeResult<T> = Result<T, TypeError>;

pub fn type_check(expr: &Expr) -> TypeResult<Type> {
    type_check_with_env(expr, &mut HashMap::new())
}

fn type_check_with_env(expr: &Expr, env: &mut TypeEnv) -> TypeResult<Type> {
    match expr {
        Expr::Int(_) => Ok(Type::Int),
        Expr::Bool(_) => Ok(Type::Bool),
        
        Expr::Var(name) => {
            env.get(name)
               .cloned()
               .ok_or_else(|| TypeError::UnboundVariable(name.clone()))
        }
        
        Expr::Lam(param, body) => {
            let param_type = Type::Int; // For now, assume all parameters are Int
            env.insert(param.clone(), param_type.clone());
            let return_type = type_check_with_env(body, env)?;
            env.remove(param);
            Ok(Type::Fun(Box::new(param_type), Box::new(return_type)))
        }
        
        Expr::App(func, arg) => {
            let func_type = type_check_with_env(func, env)?;
            let arg_type = type_check_with_env(arg, env)?;
            
            match func_type {
                Type::Fun(param_type, return_type) => {
                    if *param_type == arg_type {
                        Ok(*return_type)
                    } else {
                        Err(TypeError::TypeMismatch {
                            expected: *param_type,
                            found: arg_type,
                        })
                    }
                }
                _ => Err(TypeError::NotAFunction(func_type)),
            }
        }
        
        Expr::Let(name, expr, body) => {
            let expr_type = type_check_with_env(expr, env)?;
            env.insert(name.clone(), expr_type);
            let result = type_check_with_env(body, env);
            env.remove(name);
            result
        }
        
        Expr::BinOp(op, e1, e2) => {
            let t1 = type_check_with_env(e1, env)?;
            let t2 = type_check_with_env(e2, env)?;
            
            match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    if t1 == Type::Int && t2 == Type::Int {
                        Ok(Type::Int)
                    } else {
                        Err(TypeError::TypeMismatch {
                            expected: Type::Int,
                            found: if t1 != Type::Int { t1 } else { t2 },
                        })
                    }
                }
                BinOp::And | BinOp::Or => {
                    if t1 == Type::Bool && t2 == Type::Bool {
                        Ok(Type::Bool)
                    } else {
                        Err(TypeError::TypeMismatch {
                            expected: Type::Bool,
                            found: if t1 != Type::Bool { t1 } else { t2 },
                        })
                    }
                }
            }
        }
    }
} 