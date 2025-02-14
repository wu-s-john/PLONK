use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    #[token("let")]
    Let,
    
    #[token("in")]
    In,
    
    #[token("fun")]
    Fun,
    
    // Literals
    #[regex("[0-9]+", |lex| lex.slice().parse())]
    Int(i64),
    
    #[token("true")]
    True,
    
    #[token("false")]
    False,
    
    // Operators
    #[token("+")]
    Plus,
    
    #[token("-")]
    Minus,
    
    #[token("*")]
    Star,
    
    #[token("&&")]
    And,
    
    #[token("||")]
    Or,
    
    #[token("=")]
    Equals,
    
    #[token("->")]
    Arrow,
    
    // Delimiters
    #[token("(")]
    LParen,
    
    #[token(")")]
    RParen,
    
    // Identifiers (must come after keywords)
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),
    
    // Skip whitespace and comments
    #[regex(r"[ \t\n\r]+", logos::skip)]
    #[regex(r"//[^\n]*", logos::skip)]
    #[error]
    Error,
}

pub fn lex(input: &str) -> Vec<Token> {
    Token::lexer(input).collect()
} 