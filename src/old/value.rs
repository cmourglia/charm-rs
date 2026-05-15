use crate::bytecode::Opcode;

#[derive(Debug, Clone, Copy)]
pub enum ValueError {
    InvalidTypes {
        op: Opcode,
        lhs: Type,
        rhs: Option<Type>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
    Number,
}

#[derive(Debug, Clone, Copy)]
pub enum Value {
    Number(f64),
}

impl Value {
    pub fn add(lhs: &Value, rhs: &Value) -> Result<Value, ValueError> {
        match (lhs, rhs) {
            (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs + rhs)),
        }
    }

    pub fn sub(lhs: &Value, rhs: &Value) -> Result<Value, ValueError> {
        match (lhs, rhs) {
            (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs - rhs)),
        }
    }

    pub fn mul(lhs: &Value, rhs: &Value) -> Result<Value, ValueError> {
        match (lhs, rhs) {
            (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs * rhs)),
        }
    }

    pub fn div(lhs: &Value, rhs: &Value) -> Result<Value, ValueError> {
        match (lhs, rhs) {
            (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs / rhs)),
        }
    }
}
