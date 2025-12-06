use charm_rs::{
    lexer::tokenize,
    parser::parse,
    treewalk_interpreter::treewalk_interpret,
};

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

    treewalk_interpret(&program);
}

fn main() {
    if std::env::args().len() <= 1 {
        println!("Missing file input");
        return;
    }

    let input = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();

    interpret_program(&input);
}
