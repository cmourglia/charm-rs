use std::rc::Rc;

use crate::parser::Stmt;

//#[derive(Debug, PartialEq)]
//pub enum Cell {
//    String(String),
//}

#[derive(Debug, PartialEq, Clone)]
#[allow(unpredictable_function_pointer_comparisons)]
pub enum Value {
    Nil,
    Number(f64),
    Boolean(bool),
    //Cell(Rc<Cell>), // TODO: GC object
    // NOTE: This should probably not be handled the same way as other values at some point
    // But this will do for now
    // ...
}
