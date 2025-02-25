use super::{lexer::lex, parser};

pub fn parse_expr(input: &str) -> Box<crate::language_frontend::ast::Expr> {
    let tokens = lex(input);
    let token_triples: Vec<_> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| (i, t, i + 1))
        .collect();

    parser::ExprParser::new()
        .parse(token_triples.into_iter())
        .unwrap()
}