mod lexer;
mod parser;

fn main() {
    let prg = r#"print("Hello, World");"#;
    let mut l = lexer::Lexer::new(prg);
    let mut p = parser::Parser::new(&mut l);

    let prg = p.parse_program();

    println!("{:?}", prg);
}
