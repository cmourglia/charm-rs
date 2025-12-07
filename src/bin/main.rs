use charm_rs::{
    bytecode::{Chunk, Opcode},
    lexer::tokenize,
    parser::parse,
    treewalk_interpreter::treewalk_interpret,
    value::Value,
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

#[cfg(false)]
fn main() {
    if std::env::args().len() <= 1 {
        println!("Missing file input");
        return;
    }

    let input = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();

    interpret_program(&input);
}

fn disassemble(chunk: &Chunk, name: &str) {
    println!("== {name} ==");

    for (i, op) in chunk.code.iter().enumerate() {
        let op = match op {
            Opcode::Constant(index) => format!("OP_CONSTANT < {:?}", chunk.constants[*index]),
            Opcode::Negate => "OP_NEGATE".to_string(),
            Opcode::Return => "OP_RETURN".to_string(),
            Opcode::Add => "OP_ADD".to_string(),
            Opcode::Subtract => "OP_SUBTRACT".to_string(),
            Opcode::Multiply => "OP_MULTIPLY".to_string(),
            Opcode::Divide => "OP_DIVIDE".to_string(),
        };

        println!("{0: <04}   {1: <50}", i, op);
    }
}

#[cfg(true)]
fn main() {
    use charm_rs::vm::bytecode_interpret;

    let mut chunk = Chunk {
        code: vec![],
        constants: vec![],
    };

    chunk.push_constant(Value::Number(3.0));
    chunk.push_constant(Value::Number(2.0));
    chunk.code.push(Opcode::Negate);
    chunk.code.push(Opcode::Multiply);
    chunk.code.push(Opcode::Return);

    disassemble(&chunk, "test");

    match bytecode_interpret(&chunk) {
        Ok(()) => {}
        Err(e) => println!("Something wrong happened: {:?}", e),
    }
}
