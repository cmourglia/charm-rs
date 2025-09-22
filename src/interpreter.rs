use std::collections::HashMap;
use std::io::{self, Write};

use crate::lexer::Token;
use crate::parser::{Expr, Program, Stmt};
use crate::value::Value;

#[derive(Debug, Clone)]
pub enum RuntimeError {
    IoError(String),
}

impl From<io::Error> for RuntimeError {
    fn from(error: io::Error) -> Self {
        // TODO:
        return RuntimeError::IoError(error.to_string());
    }
}

type NativeFunction = fn(&mut Interpreter, &[Value]) -> Result<Option<Value>, RuntimeError>;

#[derive(Debug, Clone)]
enum Function {
    Native(NativeFunction),
    UserDefined { args: Vec<String>, body: Box<Stmt> },
}

#[derive(Debug)]
struct Frame {
    variables: HashMap<String, Value>, // TODO: Get rid of the copies at some point
    functions: HashMap<String, Function>,
}

#[derive(Debug)]
struct FrameStack {
    frames: Vec<Frame>,
}

impl FrameStack {
    pub fn new() -> Self {
        return FrameStack { frames: vec![] };
    }

    pub fn push_frame(&mut self) {
        self.frames.push(Frame {
            variables: HashMap::new(),
            functions: HashMap::new(),
        });
    }

    pub fn pop_frame(&mut self) {
        self.frames.pop();
    }

    // FIXME: Should this return a ref or a clone of the value ?
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        for frame in self.frames.iter().rev() {
            if let Some(value) = frame.variables.get(&name.to_string()) {
                return Some(value);
            }
        }

        return None;
    }

    pub fn declare_variable(&mut self, name: &str, value: &Value) {
        self.frames
            .last_mut()
            .unwrap()
            .variables
            .insert(name.to_string(), value.clone());
    }

    pub fn set_variable(&mut self, name: &str, value: &Value) -> Result<(), ()> {
        for frame in self.frames.iter_mut().rev() {
            if let Some(v) = frame.variables.get_mut(&name.to_string()) {
                *v = value.clone();
                return Ok(());
            }
        }

        return Err(());
    }

    // FIXME: Should this return a ref or a clone of the value ?
    pub fn get_function(&self, name: &str) -> Option<&Function> {
        for frame in self.frames.iter().rev() {
            if let Some(value) = frame.functions.get(&name.to_string()) {
                return Some(value);
            }
        }

        return None;
    }

    pub fn declare_function(&mut self, name: &str, value: &Function) {
        self.frames
            .last_mut()
            .unwrap()
            .functions
            .insert(name.to_string(), value.clone());
    }
}

fn native_print(ctx: &mut Interpreter, values: &[Value]) -> Result<Option<Value>, RuntimeError> {
    writeln!(ctx.writer, "Hello, World!")?;
    for v in values {
        writeln!(ctx.writer, "{:?}", v)?;
    }
    return Ok(None);
}

pub struct Interpreter<W: std::io::Write> {
    writer: W,
    frames: FrameStack,
}

impl<W: std::io::Write> Interpreter<W> {
    pub fn new() -> Self {
        return Self::with_writer(std::io::stdout());
    }

    #[allow(unused)]
    pub fn with_writer(writer: W) -> Self {
        return Interpreter {
            writer,
            frames: FrameStack::new(),
        };
    }

    pub fn run(&mut self, prg: &Program) {
        // Global layer
        self.frames.push_frame();

        self.frames
            .declare_function("print", &Function::Native(native_print));

        for stmt in &prg.statements {
            interpret_stmt(self, stmt);
        }
    }
}

fn interpret_stmt(ctx: &mut Interpreter, stmt: &Box<Stmt>) {
    match **stmt {
        Stmt::Expr(ref expr) => interpret_expr(ctx, &expr),
        Stmt::VarDecl {
            identifier: _,
            expr: _,
        } => todo!(),
        Stmt::FunctionDecl {
            identifier: _,
            args: _,
            body: _,
        } => todo!(),
        Stmt::Block(_) => todo!(),
        Stmt::If {
            cond: _,
            if_block: _,
            else_block: _,
        } => todo!(),
        Stmt::While { cond: _, block: _ } => todo!(),
    };
}

fn interpret_expr(ctx: &mut Interpreter, expr: &Box<Expr>) -> Value {
    match **expr {
        Expr::Number(n) => Value::Number(n),
        Expr::Boolean(b) => Value::Boolean(b),
        Expr::String(_) => todo!(),
        Expr::Identifier(_) => todo!(),

        Expr::Unary { ref rhs, ref op } => unary_expr(ctx, &rhs, op),

        Expr::Binary {
            ref lhs,
            ref rhs,
            ref op,
        } => binary_expr(ctx, &lhs, &rhs, op),

        Expr::Assignment {
            identifier: _,
            value: _,
        } => todo!(),
        Expr::Call {
            callee: _,
            arguments: _,
        } => todo!(),
    }
}

fn binary_expr(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>, op: &Token) -> Value {
    match op {
        Token::Plus => add(ctx, lhs, rhs),
        Token::Minus => sub(ctx, lhs, rhs),
        Token::Asterisk => mul(ctx, lhs, rhs),
        Token::Slash => div(ctx, lhs, rhs),
        Token::Greater => gt(ctx, lhs, rhs),
        Token::GreaterEqual => ge(ctx, lhs, rhs),
        Token::Less => lt(ctx, lhs, rhs),
        Token::LessEqual => le(ctx, lhs, rhs),
        Token::EqualEqual => eq(ctx, lhs, rhs),
        Token::BangEqual => neq(ctx, lhs, rhs),
        Token::Or => or(ctx, lhs, rhs),
        Token::And => and(ctx, lhs, rhs),
        _ => unreachable!(),
    }
}

fn add(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs + rhs),
        _ => unreachable!(),
    }
}

fn sub(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs - rhs),
        _ => unreachable!(),
    }
}

fn mul(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs * rhs),
        _ => unreachable!(),
    }
}

fn div(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs / rhs),
        _ => unreachable!(),
    }
}

fn and(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);

    if let Value::Boolean(lhs) = lhs {
        if !lhs {
            return Value::Boolean(false);
        }

        let rhs = interpret_expr(ctx, rhs);

        if let Value::Boolean(rhs) = rhs {
            Value::Boolean(lhs && rhs)
        } else {
            unreachable!()
        }
    } else {
        unreachable!();
    }
}

fn or(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);

    if let Value::Boolean(lhs) = lhs {
        if lhs {
            return Value::Boolean(true);
        }

        let rhs = interpret_expr(ctx, rhs);

        if let Value::Boolean(rhs) = rhs {
            Value::Boolean(lhs || rhs)
        } else {
            unreachable!()
        }
    } else {
        unreachable!();
    }
}

fn lt(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs < rhs),
        _ => unreachable!(),
    }
}

fn le(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs <= rhs),
        _ => unreachable!(),
    }
}

fn gt(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs > rhs),
        _ => unreachable!(),
    }
}

fn ge(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs >= rhs),
        _ => unreachable!(),
    }
}

fn eq(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    return Value::Boolean(lhs == rhs);
}

fn neq(ctx: &mut Interpreter, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    return Value::Boolean(lhs != rhs);
}

fn unary_expr(ctx: &mut Interpreter, right: &Box<Expr>, op: &Token) -> Value {
    let right_value = interpret_expr(ctx, right);

    match op {
        Token::Minus => {
            if let Value::Number(v) = right_value {
                Value::Number(-v)
            } else {
                unreachable!()
            }
        }
        Token::Not => {
            if let Value::Boolean(b) = right_value {
                Value::Boolean(!b)
            } else {
                unreachable!()
            }
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod program_tests {
    use std::{
        io::{self, Write},
        sync::{Arc, Mutex},
    };

    use crate::{lexer::tokenize, parser::parse};

    use super::*;

    #[derive(Clone)]
    struct TestWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl TestWriter {
        fn new() -> Self {
            return Self {
                buffer: Arc::new(Mutex::new(vec![])),
            };
        }

        fn get_output(&self) -> String {
            let buffer = self.buffer.lock().unwrap();
            return String::from_utf8(buffer.clone()).unwrap();
        }
    }

    impl Write for TestWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut buffer = self.buffer.lock().unwrap();
            buffer.extend_from_slice(buf);
            return Ok(buf.len());
        }

        fn flush(&mut self) -> io::Result<()> {
            return Ok(());
        }
    }

    fn test_program_output(input: &str, expected: &str) {
        let (tokens, errors) = tokenize(input);
        assert_eq!(errors, vec![]);

        let program = parse(tokens);
        assert_eq!(program.errors, vec![]);
        assert_eq!(program.statements.len(), 1);

        let writer = TestWriter::new();
        let mut interpreter = Interpreter::with_writer(writer.clone());

        interpreter.run(&program);

        let output = writer.get_output();

        assert_eq!(expected, output);
    }

    #[test]
    fn testtest() {
        test_program_output("print(42);", "");
    }
}

#[cfg(test)]
mod expression_tests {
    use crate::{lexer::tokenize, parser::parse};

    use super::*;

    fn test_eval_expression(input: &str, expected: Value) {
        let (tokens, errors) = tokenize(input);
        assert_eq!(errors, vec![]);

        let program = parse(tokens);
        assert_eq!(program.errors, vec![]);
        assert_eq!(program.statements.len(), 1);

        let mut ctx = Interpreter::new();

        match *program.statements[0] {
            Stmt::Expr(ref expr) => {
                let value = interpret_expr(&mut ctx, expr);
                assert_eq!(value, expected);
            }
            _ => assert!(false),
        }
    }

    #[test]
    fn number() {
        test_eval_expression("2;", Value::Number(2.0));
    }

    #[test]
    fn boolean() {
        test_eval_expression("true;", Value::Boolean(true));
        test_eval_expression("false;", Value::Boolean(false));
    }

    #[test]
    fn unary() {
        test_eval_expression("-42.0;", Value::Number(-42.0));
        test_eval_expression("--42.0;", Value::Number(42.0));
        test_eval_expression("not true;", Value::Boolean(false));
        test_eval_expression("not not not not false;", Value::Boolean(false));
    }

    #[test]
    fn binary_arithmetic() {
        test_eval_expression("3 + 2;", Value::Number(5.0));
        test_eval_expression("4 - 1;", Value::Number(3.0));
        test_eval_expression("10.5 * 4;", Value::Number(42.0));
        test_eval_expression("336 / 8;", Value::Number(42.0));

        test_eval_expression("3 + 5 * 4 - 12 / 2;", Value::Number(17.0));

        test_eval_expression("2 * 3 + 4;", Value::Number(10.0));
        test_eval_expression("2 * (3 + 4);", Value::Number(14.0));
    }

    #[test]
    fn binary_logic() {
        test_eval_expression("true and false;", Value::Boolean(false));
        test_eval_expression("false and 42.0;", Value::Boolean(false));

        test_eval_expression("true or 1337;", Value::Boolean(true));
        test_eval_expression("false or true;", Value::Boolean(true));
    }

    #[test]
    fn comparisons() {
        test_eval_expression("1 < 2;", Value::Boolean(true));
        test_eval_expression("2 <= 1;", Value::Boolean(false));
        test_eval_expression("4 > 3.99;", Value::Boolean(true));
        test_eval_expression("3.99 >= 4.01;", Value::Boolean(false));
        test_eval_expression("true == false;", Value::Boolean(false));
        test_eval_expression("true != false;", Value::Boolean(true));
    }
}
