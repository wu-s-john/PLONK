use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::language_frontend::ast::{Expr, BinOp, Type};

#[derive(Debug)]
pub enum TypeError {
    UnboundVariable(String),
    OccursCheckFailed,
    UnificationFail(Type, Type),
}

static FRESH_VAR_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn fresh_tvar() -> Type {
    let id = FRESH_VAR_COUNTER.fetch_add(1, Ordering::SeqCst);
    Type::TVar(id)
}

/// A substitution maps from a type-variable ID to a concrete Type.
/// Example:  α2 -> Int, α3 -> Fun(TInt, TBool)
type Substitution = HashMap<usize, Type>;

/// Apply a substitution to a Type (recursively).
fn apply_subst(ty: &Type, subst: &Substitution) -> Type {
    match ty {
        Type::Int => Type::Int,
        Type::Bool => Type::Bool,
        Type::Fun(t1, t2) => {
            Type::Fun(
                Box::new(apply_subst(t1, subst)),
                Box::new(apply_subst(t2, subst)),
            )
        }
        Type::TVar(id) => {
            if let Some(t) = subst.get(id) {
                // We must recursively apply the substitution to what it maps to,
                // in case t itself contains other TVars
                apply_subst(t, subst)
            } else {
                Type::TVar(*id)
            }
        }
    }
}

/// Apply a substitution to every binding in the environment.
fn apply_subst_env(env: &mut HashMap<String, Type>, subst: &Substitution) {
    for ty in env.values_mut() {
        *ty = apply_subst(ty, subst);
    }
}

/// Compose two substitutions s2 ∘ s1, meaning "apply s1 first, then s2".
/// In code, that typically means:
///   1. For each (α -> t) in s1, apply s2 to t.
///   2. Take (α -> t') pairs from s2 as well (these override if α appears in both).
fn compose_subst(s2: &Substitution, s1: &Substitution) -> Substitution {
    let mut result = HashMap::new();

    // First, map α -> apply_subst(t, s2) for each (α -> t) in s1
    for (var, t) in s1 {
        let new_t = apply_subst(t, s2);
        result.insert(*var, new_t);
    }

    // Then, take everything from s2 (with override)
    for (var, t) in s2 {
        result.insert(*var, t.clone());
    }

    result
}

/// Check if TVar(uid) occurs inside type `ty` to avoid α = Fun(…, α)-style infinite types.
fn occurs_check(uid: usize, ty: &Type, subst: &Substitution) -> bool {
    let ty_applied = apply_subst(ty, subst);
    match ty_applied {
        Type::TVar(id) => id == uid,
        Type::Fun(t1, t2) => occurs_check(uid, &t1, subst) || occurs_check(uid, &t2, subst),
        _ => false,
    }
}

/// Unify two types, returning a substitution that makes them equal (if possible).
fn unify(t1: &Type, t2: &Type) -> Result<Substitution, TypeError> {
    match (t1, t2) {
        (Type::Int, Type::Int) => Ok(HashMap::new()),
        (Type::Bool, Type::Bool) => Ok(HashMap::new()),
        (Type::Fun(a1, a2), Type::Fun(b1, b2)) => {
            // Unify a1 with b1
            let s1 = unify(a1, b1)?;
            // Apply s1 to a2 and b2
            let a2_sub = apply_subst(a2, &s1);
            let b2_sub = apply_subst(b2, &s1);
            // Then unify the results
            let s2 = unify(&a2_sub, &b2_sub)?;
            // Finally compose the two
            Ok(compose_subst(&s2, &s1))
        }
        (Type::TVar(id1), Type::TVar(id2)) if id1 == id2 => {
            // α unified with α is trivially empty
            Ok(HashMap::new())
        }
        (Type::TVar(uid), other) => {
            // If α occurs in other, no solution
            // e.g. unify(α, α->Int) would be infinite
            if occurs_check(*uid, other, &HashMap::new()) {
                return Err(TypeError::OccursCheckFailed);
            }
            // Subst: α -> other
            let mut s = HashMap::new();
            s.insert(*uid, other.clone());
            Ok(s)
        }
        (other, Type::TVar(uid)) => {
            // symmetric case
            if occurs_check(*uid, other, &HashMap::new()) {
                return Err(TypeError::OccursCheckFailed);
            }
            let mut s = HashMap::new();
            s.insert(*uid, other.clone());
            Ok(s)
        }
        _ => {
            // e.g. unify(Int, Bool) => error
            Err(TypeError::UnificationFail(t1.clone(), t2.clone()))
        }
    }
}

/// We store an environment mapping VarName -> Type (possibly containing TVars).
type TypeEnv = HashMap<String, Type>;

/// The W algorithm (monomorphic). Returns (S, τ).
/// - S is the substitution found so far
/// - τ is the type of `expr` after applying S
fn infer_expr(expr: &Expr, env: &mut TypeEnv) -> Result<(Substitution, Type), TypeError> {
    match expr {
        Expr::Int(_) => Ok((HashMap::new(), Type::Int)),
        Expr::Bool(_) => Ok((HashMap::new(), Type::Bool)),

        Expr::Var(name) => {
            if let Some(ty) = env.get(name).cloned() {
                // No new substitution
                Ok((HashMap::new(), ty))
            } else {
                Err(TypeError::UnboundVariable(name.clone()))
            }
        }

        Expr::Lam(param, body) => {
            // Introduce a fresh type var for param
            let param_ty = fresh_tvar();
            let old = env.insert(param.clone(), param_ty.clone());
            let (s1, body_ty) = infer_expr(body, env)?;
            // remove the param from env (restore old)
            if let Some(old_ty) = old {
                env.insert(param.clone(), old_ty);
            } else {
                env.remove(param);
            }
            // apply s1 to param_ty (it might have been unified by uses)
            let param_ty_applied = apply_subst(&param_ty, &s1);
            let fun_ty = Type::Fun(Box::new(param_ty_applied), Box::new(body_ty));

            Ok((s1, fun_ty))
        }

        Expr::App(func, arg) => {
            // infer foo
            let (s1, func_ty) = infer_expr(func, env)?;
            // apply s1 to env before inferring argument
            apply_subst_env(env, &s1);

            // infer arg (in updated env)
            let (s2, arg_ty) = infer_expr(arg, env)?;
            // compose s2 with s1
            let s12 = compose_subst(&s2, &s1);

            // fresh type var for the result
            let result_ty = fresh_tvar();
            // unify func_ty with (arg_ty -> result_ty), applying s12 to them first

            let func_ty_sub = apply_subst(&func_ty, &s2);
            let want_ty = Type::Fun(Box::new(arg_ty.clone()), Box::new(result_ty.clone()));

            let s3 = unify(&func_ty_sub, &want_ty)?;
            // final substitution is s3 ∘ s2 ∘ s1
            let s_final = compose_subst(&s3, &s12);

            let result_ty_final = apply_subst(&result_ty, &s_final);
            Ok((s_final, result_ty_final))
        }

        Expr::Let(name, e1, body) => {
            // infer e1
            let (s1, ty_e1) = infer_expr(e1, env)?;
            // apply s1 to env
            apply_subst_env(env, &s1);

            // put name -> ty_e1 in env
            let old = env.insert(name.clone(), ty_e1);
            // infer body
            let (s2, body_ty) = infer_expr(body, env)?;
            // remove binding or restore old
            if let Some(old_ty) = old {
                env.insert(name.clone(), old_ty);
            } else {
                env.remove(name);
            }

            // compose s2 with s1
            let s_final = compose_subst(&s2, &s1);
            let body_ty_final = apply_subst(&body_ty, &s_final);

            Ok((s_final, body_ty_final))
        }

        Expr::BinOp(op, e1, e2) => {
            let (s1, t1) = infer_expr(e1, env)?;
            apply_subst_env(env, &s1);
            let (s2, t2) = infer_expr(e2, env)?;
            let s12 = compose_subst(&s2, &s1);

            match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    // unify t1 with Int and unify t2 with Int
                    let s3 = unify(&apply_subst(&t1, &s12), &Type::Int)?;
                    let s4 = unify(&apply_subst(&t2, &s3), &Type::Int)?;
                    let s_final = compose_subst(&s4, &compose_subst(&s3, &s12));
                    let ty = Type::Int;
                    Ok((s_final, ty))
                }

                BinOp::And | BinOp::Or => {
                    // unify t1 with Bool, unify t2 with Bool
                    let s3 = unify(&apply_subst(&t1, &s12), &Type::Bool)?;
                    let s4 = unify(&apply_subst(&t2, &s3), &Type::Bool)?;
                    let s_final = compose_subst(&s4, &compose_subst(&s3, &s12));
                    let ty = Type::Bool;
                    Ok((s_final, ty))
                }
            }
        }
    }
}

/// Public entry point: tries to infer the type of `expr` and returns the fully substituted type.
pub fn type_check(expr: &Expr) -> Result<Type, TypeError> {
    let mut env = HashMap::new();
    let (subst, ty) = infer_expr(expr, &mut env)?;
    let final_ty = apply_subst(&ty, &subst);
    Ok(final_ty)
} 