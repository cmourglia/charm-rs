//#[derive(Debug, PartialEq)]
//pub enum Cell {
//    String(String),
//}

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Nil,
    Number(f64),
    Boolean(bool),
    // NOTE: This should probably not be handled the same way as other values at some point
    // But this will do for now
    // ...
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Boolean(b) => *b,
            _ => false,
        }
    }
}
