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

type NativeFunction = fn(&mut Context, &[Value]) -> Result<Option<Value>, RuntimeError>;

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

pub struct Context {
    writer: Box<dyn std::io::Write>,
    frames: FrameStack,
}

impl Context {
    pub fn new() -> Self {
        return Self::with_writer(Box::new(std::io::stdout()));
    }

    #[allow(unused)]
    pub fn with_writer(writer: Box<dyn std::io::Write>) -> Context {
        return Context {
            writer,
            frames: FrameStack::new(),
        };
    }
}

fn native_print(ctx: &mut Context, values: &[Value]) -> Result<Option<Value>, RuntimeError> {
    let mut first = true;

    for v in values {
        if !first {
            write!(ctx.writer, " ")?;
        }

        match v {
            Value::Nil => write!(ctx.writer, "<nil>")?,
            Value::Number(n) => write!(ctx.writer, "{}", n)?,
            Value::Boolean(b) => write!(ctx.writer, "{}", b)?,
        }

        first = false;
    }

    return Ok(None);
}

pub fn interpret(prg: &Program) {
    let mut ctx = Context::new();

    interpret_with_context(&mut ctx, prg);
}

fn interpret_with_context(ctx: &mut Context, prg: &Program) {
    // Global layer
    ctx.frames.push_frame();

    ctx.frames
        .declare_function("print", &Function::Native(native_print));

    for stmt in &prg.statements {
        interpret_stmt(ctx, stmt);
    }
}

fn interpret_stmt(ctx: &mut Context, stmt: &Box<Stmt>) {
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

fn interpret_expr(ctx: &mut Context, expr: &Box<Expr>) -> Value {
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
            ref callee,
            ref arguments,
        } => {
            let function = match ctx.frames.get_function(&callee) {
                Some(function) => function.clone(),
                None => todo!("error"),
            };

            let mut args = Vec::new();
            for arg in arguments {
                args.push(interpret_expr(ctx, arg));
            }

            ctx.frames.push_frame();

            let result = match function {
                Function::Native(function) => call_native_function(ctx, &function, &args),
                Function::UserDefined { args, body } => Value::Nil,
            };

            ctx.frames.pop_frame();

            result
        }
    }
}

fn call_native_function(
    ctx: &mut Context,
    function: &NativeFunction,
    arguments: &[Value],
) -> Value {
    function(ctx, arguments);

    return Value::Nil;
}

fn binary_expr(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>, op: &Token) -> Value {
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

fn add(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs + rhs),
        _ => unreachable!(),
    }
}

fn sub(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs - rhs),
        _ => unreachable!(),
    }
}

fn mul(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs * rhs),
        _ => unreachable!(),
    }
}

fn div(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs / rhs),
        _ => unreachable!(),
    }
}

fn and(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
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

fn or(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
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

fn lt(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs < rhs),
        _ => unreachable!(),
    }
}

fn le(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs <= rhs),
        _ => unreachable!(),
    }
}

fn gt(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs > rhs),
        _ => unreachable!(),
    }
}

fn ge(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs >= rhs),
        _ => unreachable!(),
    }
}

fn eq(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    return Value::Boolean(lhs == rhs);
}

fn neq(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(ctx, lhs);
    let rhs = interpret_expr(ctx, rhs);

    return Value::Boolean(lhs != rhs);
}

fn unary_expr(ctx: &mut Context, right: &Box<Expr>, op: &Token) -> Value {
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
        let mut ctx = Context::with_writer(Box::new(writer.clone()));

        interpret_with_context(&mut ctx, &program);

        let output = writer.get_output();

        println!("{}", output);

        assert_eq!(expected, output);
    }

    #[test]
    fn print_call() {
        test_program_output("print(42);", "42");
        test_program_output("print(42, 43, 44);", "42 43 44");
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

        let mut ctx = Context::new();

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
