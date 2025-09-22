use crate::{interpreter::Interpreter, lexer::tokenize, parser::parse};

mod interpreter;
mod lexer;
mod parser;
mod value;

pub fn interpret_program(input: &str) {
    let (tokens, lex_errors) = tokenize(input);
    if !lex_errors.is_empty() {
        println!("Tokenize errors :");
        for e in lex_errors {
            println!("  {:?}", e);
        }
        return;
    }

    let program = parse(tokens);

    let mut interpreter = Interpreter::new();
    interpreter.run(&program);
}
