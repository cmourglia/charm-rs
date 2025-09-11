mod interpreter;
mod lexer;
mod parser;
mod value;

pub fn interpret_program(input: &str) {
    let program = parser::parse_program(input);
    interpreter::interpret_program(&program);
}
