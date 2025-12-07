use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    Constant(usize),
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Return,
}

#[derive(Debug)]
pub struct Chunk {
    pub code: Vec<Opcode>,
    pub constants: Vec<Value>,
}

impl Chunk {
    pub fn push_constant(&mut self, value: Value) {
        let constant_id = self.constants.len();
        self.constants.push(value);
        self.code.push(Opcode::Constant(constant_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
