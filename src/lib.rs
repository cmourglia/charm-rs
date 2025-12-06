use crate::lexer::lexer::tokenize;
use crate::parser::parser::parse;
use crate::treewalk::interpreter::interpret;

mod compiler;
mod lexer;
mod parser;
mod treewalk;

pub fn interpret_program(input: &str) {
    let (tokens, lex_errors) = tokenize(input);
    if !lex_errors.is_empty() {
        println!("Tokenize errors:");
        for e in lex_errors {
            println!(" {:?}", e);
        }
        return;
    }

    let program = parse(tokens);

    if !program.errors.is_empty() {
        println!("Parse errors:");
        for e in program.errors {
            println!(" {:?}", e);
        }
        return;
    }

    interpret(&program);
}

fn variant_eq<T>(a: &T, b: &T) -> bool {
    return std::mem::discriminant(a) == std::mem::discriminant(b);
}
