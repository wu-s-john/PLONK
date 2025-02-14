use ark_ff::Field;
use ark_std::{test_rng, UniformRand};
use ark_bn254::Fr;

mod ast;
mod execution_trace;
mod offset_table;
mod permutation_polynomial;
mod polynomial_utils;
mod language_frontend;

use std::io::{self, Write};
use crate::language_frontend::{ast as lang_ast, lexer, type_checker, evaluator};
use crate::language_frontend::parser;

#[derive(Debug)]
enum InterpreterError {
    ParseError(String),
    TypeError(type_checker::TypeError),
    EvalError(evaluator::EvalError),
}

fn process_input(input: &str) -> Result<(lang_ast::Expr, lang_ast::Type, lang_ast::Value), InterpreterError> {
    // Tokenize
    let tokens = lexer::lex(input);
    
    // Parse using LALRPOP parser
    let ast = *parser::ExprParser::new()
        .parse(tokens.iter().enumerate().map(|(i, t)| (i, t.clone(), i + 1)))
        .map_err(|e| InterpreterError::ParseError(format!("{:?}", e)))?;
    
    // Type check
    let typ = type_checker::type_check(&ast)
        .map_err(InterpreterError::TypeError)?;
    
    // Evaluate
    let value = evaluator::eval(&ast)
        .map_err(InterpreterError::EvalError)?;
    
    Ok((ast, typ, value))
}

fn main() -> io::Result<()> {
    println!("Simple Lambda Calculus Interpreter");
    println!("Enter expressions (Ctrl+D to exit):");
    
    let mut input = String::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    
    loop {
        print!("> ");
        stdout.flush()?;
        
        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }
        
        match process_input(&input) {
            Ok((ast, typ, value)) => {
                println!("AST: {:?}", ast);
                println!("Type: {}", typ);
                println!("Value: {:?}", value);
            }
            Err(e) => println!("Error: {:?}", e),
        }
    }
    
    Ok(())
}