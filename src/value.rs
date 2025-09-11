#[derive(Debug, PartialEq)]
pub enum Value {
    Nil,
    Number(f64),
    Boolean(bool),
    // ...
}
