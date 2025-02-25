#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),               // Integer literals
    Bool(bool),             // Boolean literals
    Var(String),           // Variables
    Lam(String, Box<Expr>), // Lambda abstractions (functions)
    App(Box<Expr>, Box<Expr>), // Function application
    Let(String, Box<Expr>, Box<Expr>), // Let bindings
    BinOp(BinOp, Box<Expr>, Box<Expr>), // Binary operations
    If(Box<Expr>, Box<Expr>, Box<Expr>), // If-then-else expressions
}

#[derive(Debug, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Eq,  // Added equality operator
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Bool,
    Fun(Box<Type>, Box<Type>), // Function type T1 -> T2
    TVar(usize),               // Type variable for type inference
}

// Runtime values
#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Closure(String, Box<Expr>, Environment), // Closure with captured environment
}

use std::collections::HashMap;
pub type Environment = HashMap<String, Value>;

// Pretty printing implementation for Expr
impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Int(n) => write!(f, "{}", n),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::Var(x) => write!(f, "{}", x),
            Expr::Lam(param, body) => write!(f, "fun {} -> {}", param, body),
            Expr::App(func, arg) => write!(f, "({} {})", func, arg),
            Expr::Let(name, expr, body) => write!(f, "let {} = {} in {}", name, expr, body),
            Expr::BinOp(op, e1, e2) => {
                let op_str = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::And => "&&",
                    BinOp::Or => "||",
                    BinOp::Eq => "==",  // Added equality operator string
                };
                write!(f, "({} {} {})", e1, op_str, e2)
            }
            Expr::If(cond, then, else_expr) => write!(f, "if {} then {} else {}", cond, then, else_expr),
        }
    }
}

// Pretty printing for Type
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
            Type::Fun(t1, t2) => write!(f, "({} -> {})", t1, t2),
            Type::TVar(id) => write!(f, "α{}", id),  // Use α for type variables
        }
    }
}

// Pretty printing for Value
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Closure(param, body, _) => write!(f, "<closure: fun {} -> {}>", param, body),
        }
    }
} 