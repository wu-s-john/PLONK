use std::io::{self, Write};
use crate::language_frontend::{ast as lang_ast, lexer, type_checker, evaluator};
use crate::language_frontend::parser;

mod plonk_circuit;
mod execution_trace;
mod grand_product;
mod polynomial_utils;
mod language_frontend;
mod ast_to_plonk;
mod union_find;

#[derive(Debug)]
enum InterpreterError {
    ParseError(String),
    TypeError(type_checker::TypeError),
    EvalError(evaluator::EvalError),
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpreterError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            InterpreterError::TypeError(err) => write!(f, "Type error: {:?}", err),
            InterpreterError::EvalError(err) => write!(f, "Runtime error: {:?}", err),
        }
    }
}

fn process_input(input: &str) -> Result<(lang_ast::Expr, lang_ast::Type, lang_ast::Value), InterpreterError> {
    // Skip empty input or whitespace-only input
    if input.trim().is_empty() {
        return Err(InterpreterError::ParseError("Empty input".to_string()));
    }

    // Tokenize
    let tokens = lexer::lex(input);
    if tokens.is_empty() {
        return Err(InterpreterError::ParseError("No valid tokens found".to_string()));
    }
    
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

fn print_help() {
    println!("\nAvailable commands:");
    println!("  :help, :h     - Show this help message");
    println!("  :quit, :q     - Exit the interpreter");
    println!("  :type, :t     - Show only the type of the expression");
    println!("\nExample expressions:");
    println!("  42                          - Integer literal");
    println!("  true                        - Boolean literal");
    println!("  let x = 5 in x + 3         - Let binding");
    println!("  fun x -> x + 1             - Function definition");
    println!("  (fun x -> x + 1) 5         - Function application");
    println!("  let add = fun x -> fun y -> x + y in add 2 3  - Higher-order function\n");
}

fn main() -> io::Result<()> {
    println!("Simple Lambda Calculus Interpreter");
    println!("Type :help or :h for help, :quit or :q to exit");
    println!();
    
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

        let trimmed = input.trim();
        
        // Handle special commands
        match trimmed {
            ":quit" | ":q" => break,
            ":help" | ":h" => {
                print_help();
                continue;
            }
            ":type" | ":t" => {
                println!("Usage: Type an expression after :type or :t");
                continue;
            }
            "" => continue,  // Skip empty lines
            _ => {
                // Check if it's a type query
                let (command, expr) = if trimmed.starts_with(":type ") || trimmed.starts_with(":t ") {
                    let space_idx = trimmed.find(' ').unwrap();
                    (&trimmed[..space_idx], &trimmed[space_idx + 1..])
                } else {
                    ("", trimmed)
                };

                match process_input(expr) {
                    Ok((ast, typ, value)) => {
                        if command.starts_with(":t") {
                            println!("{}", typ);
                        } else {
                            println!("AST: {}", ast);
                            println!("Type: {}", typ);
                            println!("Value: {}", value);
                        }
                    }
                    Err(e) => println!("{}", e),
                }
            }
        }
    }
    
    println!("\nGoodbye!");
    Ok(())
}