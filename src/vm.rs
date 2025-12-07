use crate::bytecode::{Chunk, Opcode};
use crate::value::{Type, Value, ValueError};

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    IoError(String),
    VariableNotDeclared(String),
    InvalidType(String),
    //TypeMismatch { old_value: Type, new_value: Type },
    InvalidNumberOfParameters,
    InvalidOperation,
    InvalidOperationTypes {
        op: Opcode,
        lhs: Type,
        rhs: Option<Type>,
    },
    EmptyStack,
}

impl From<std::io::Error> for RuntimeError {
    fn from(error: std::io::Error) -> Self {
        return RuntimeError::IoError(error.to_string());
    }
}

impl From<ValueError> for RuntimeError {
    fn from(error: ValueError) -> Self {
        match error {
            ValueError::InvalidTypes { op, lhs, rhs } => {
                RuntimeError::InvalidOperationTypes { op, lhs, rhs }
            }
        }
    }
}

pub fn bytecode_interpret(chunk: &Chunk) -> Result<(), RuntimeError> {
    interpret_with_writer(Box::new(std::io::stdout()), chunk)
}

fn interpret_with_writer(
    writer: Box<dyn std::io::Write>,
    chunk: &Chunk,
) -> Result<(), RuntimeError> {
    let mut vm = Vm {
        writer,
        chunk,
        ip: 0,
        stack: vec![],
    };

    vm.run()
}

struct Vm<'a> {
    writer: Box<dyn std::io::Write>,
    chunk: &'a Chunk,
    ip: usize,
    stack: Vec<Value>,
}

impl<'a> Vm<'a> {
    fn run(&mut self) -> Result<(), RuntimeError> {
        loop {
            let op = self.chunk.code[self.ip];

            match op {
                Opcode::Return => {
                    let top = self.pop()?;
                    writeln!(self.writer, "{:?}", top)?;

                    return Ok(());
                }
                Opcode::Constant(index) => {
                    let constant = &self.chunk.constants[index];
                    self.stack.push(constant.clone());
                }
                Opcode::Negate => match self.pop()? {
                    Value::Number(n) => self.stack.push(Value::Number(-n)),
                },

                Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide => {
                    self.binary_op(op)?;
                }

                _ => todo!(),
            }

            self.ip += 1;
        }
    }

    fn pop(&mut self) -> Result<Value, RuntimeError> {
        if let Some(value) = self.stack.pop() {
            Ok(value)
        } else {
            Err(RuntimeError::EmptyStack)
        }
    }

    fn binary_op(&mut self, opcode: Opcode) -> Result<(), RuntimeError> {
        let rhs = self.pop()?;
        let lhs = self.pop()?;

        let res = match opcode {
            Opcode::Add => Value::add(&lhs, &rhs)?,
            Opcode::Subtract => Value::sub(&lhs, &rhs)?,
            Opcode::Multiply => Value::mul(&lhs, &rhs)?,
            Opcode::Divide => Value::div(&lhs, &rhs)?,
            _ => return Err(RuntimeError::InvalidOperation),
        };

        self.stack.push(res);

        Ok(())
    }
}
