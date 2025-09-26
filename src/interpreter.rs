use std::collections::HashMap;
use std::io::{self, Write};

use crate::lexer::Token;
use crate::parser::{Expr, Program, Stmt};
use crate::value::Value;
use crate::variant_eq;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    IoError(String),
    VariableNotDeclared(String),
    FunctionNotDeclared(String),
    TypeMismatch { old_value: Value, new_value: Value },
    InvalidNumberOfParameters,
}

impl From<io::Error> for RuntimeError {
    fn from(error: io::Error) -> Self {
        return RuntimeError::IoError(error.to_string());
    }
}

type NativeFunction = fn(&mut Context, &[Value]) -> Result<FlowControl, RuntimeError>;

enum FlowControl {
    None,
    Return(Value),
}

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
    pub fn get_variable(&self, name: &str) -> Result<Value, RuntimeError> {
        for frame in self.frames.iter().rev() {
            if let Some(value) = frame.variables.get(&name.to_string()) {
                return Ok(value.clone());
            }
        }

        return Err(RuntimeError::VariableNotDeclared(name.to_string()));
    }

    pub fn declare_variable(&mut self, name: &str, value: Value) {
        self.frames
            .last_mut()
            .unwrap()
            .variables
            .insert(name.to_string(), value);
    }

    pub fn set_variable(&mut self, name: &str, value: Value) -> Result<(), RuntimeError> {
        for frame in self.frames.iter_mut().rev() {
            if let Some(v) = frame.variables.get_mut(&name.to_string()) {
                *v = value;
                return Ok(());
            }
        }

        return Err(RuntimeError::VariableNotDeclared(name.to_string()));
    }

    // FIXME: Should this return a ref or a clone of the value ?
    pub fn get_function(&self, name: &str) -> Result<Function, RuntimeError> {
        for frame in self.frames.iter().rev() {
            if let Some(value) = frame.functions.get(&name.to_string()) {
                return Ok(value.clone());
            }
        }

        return Err(RuntimeError::FunctionNotDeclared(name.to_string()));
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
    epoch: std::time::Instant,
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
            epoch: std::time::Instant::now(),
        };
    }
}

fn native_print(ctx: &mut Context, values: &[Value]) -> Result<FlowControl, RuntimeError> {
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

    writeln!(ctx.writer, "")?;

    return Ok(FlowControl::None);
}

fn native_time(ctx: &mut Context, _: &[Value]) -> Result<FlowControl, RuntimeError> {
    let now = std::time::Instant::now();

    let elapsed = now.duration_since(ctx.epoch).as_nanos() as f64;

    return Ok(FlowControl::Return(Value::Number(elapsed)));
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
    ctx.frames
        .declare_function("time", &Function::Native(native_time));

    for stmt in &prg.statements {
        match interpret_stmt(ctx, stmt) {
            Ok(_) => {}
            Err(err) => {
                println!("Runtime error: {:?}", err);
                break;
            }
        }
    }
}

fn interpret_stmt(ctx: &mut Context, stmt: &Box<Stmt>) -> Result<FlowControl, RuntimeError> {
    let result = match **stmt {
        Stmt::Expr(ref expr) => {
            interpret_expr(ctx, expr)?;
            FlowControl::None
        }

        Stmt::Block(ref stmts) => {
            ctx.frames.push_frame();

            let mut block_result = FlowControl::None;

            for stmt in stmts {
                block_result = interpret_stmt(ctx, stmt)?;

                match &block_result {
                    FlowControl::Return(_) => break,
                    FlowControl::None => {}
                }
            }

            ctx.frames.pop_frame();

            block_result
        }

        Stmt::VarDecl {
            ref identifier,
            ref expr,
        } => {
            let value = match expr {
                Some(expr) => interpret_expr(ctx, expr)?,
                None => Value::Nil,
            };

            ctx.frames.declare_variable(&identifier, value);

            FlowControl::None
        }

        Stmt::FunctionDecl {
            ref identifier,
            ref args,
            ref body,
        } => {
            let function = Function::UserDefined {
                args: args.clone(),
                body: body.clone(),
            };

            ctx.frames.declare_function(&identifier, &function);

            FlowControl::None
        }

        Stmt::If {
            ref cond,
            ref if_block,
            ref else_block,
        } => {
            let flow_control = if interpret_expr(ctx, cond)?.is_truthy() {
                interpret_stmt(ctx, if_block)?
            } else if let Some(else_block) = else_block {
                interpret_stmt(ctx, else_block)?
            } else {
                FlowControl::None
            };

            flow_control
        }

        Stmt::While {
            ref cond,
            ref block,
        } => interpret_while(ctx, cond, block)?,

        Stmt::Return(ref expr) => {
            if let Some(expr) = expr {
                FlowControl::Return(interpret_expr(ctx, expr)?)
            } else {
                FlowControl::Return(Value::Nil)
            }
        }
    };

    return Ok(result);
}

fn interpret_while(
    ctx: &mut Context,
    cond: &Box<Expr>,
    body: &Box<Stmt>,
) -> Result<FlowControl, RuntimeError> {
    while interpret_expr(ctx, cond)?.is_truthy() {
        match interpret_stmt(ctx, body)? {
            FlowControl::None => continue,
            // TODO: Handle continue
            // TODO: Handle break
            FlowControl::Return(val) => return Ok(FlowControl::Return(val)),
        }
    }

    return Ok(FlowControl::None);
}

fn interpret_expr(ctx: &mut Context, expr: &Box<Expr>) -> Result<Value, RuntimeError> {
    match **expr {
        Expr::Number(n) => Ok(Value::Number(n)),
        Expr::Boolean(b) => Ok(Value::Boolean(b)),
        Expr::String(_) => todo!(),

        Expr::Identifier(ref name) => Ok(ctx.frames.get_variable(&name)?),

        Expr::Unary { ref rhs, ref op } => Ok(unary_expr(ctx, &rhs, op)?),

        Expr::Binary {
            ref lhs,
            ref rhs,
            ref op,
        } => Ok(binary_expr(ctx, &lhs, &rhs, op)?),

        Expr::Assignment {
            ref identifier,
            ref value,
        } => {
            let new_value = interpret_expr(ctx, value)?;

            let old_value = ctx.frames.get_variable(&identifier)?;

            if old_value != Value::Nil && !variant_eq(&old_value, &new_value) {
                return Err(RuntimeError::TypeMismatch {
                    old_value,
                    new_value,
                });
            }

            ctx.frames.set_variable(&identifier, new_value.clone())?;

            Ok(new_value)
        }

        Expr::Call {
            ref callee,
            ref arguments,
        } => {
            let function = ctx.frames.get_function(&callee)?;

            let mut args = Vec::new();
            for arg in arguments {
                args.push(interpret_expr(ctx, arg)?);
            }

            ctx.frames.push_frame();

            let result = match function {
                Function::Native(function) => function(ctx, &args)?,
                Function::UserDefined {
                    args: ref params,
                    ref body,
                } => call_user_function(ctx, &params, &args, body)?,
            };

            ctx.frames.pop_frame();

            match result {
                FlowControl::Return(retval) => Ok(retval),
                _ => Ok(Value::Nil),
            }
        }
    }
}

fn call_user_function(
    ctx: &mut Context,
    params: &[String],
    args: &[Value],
    body: &Box<Stmt>,
) -> Result<FlowControl, RuntimeError> {
    let arity = args.len();

    if params.len() != arity {
        return Err(RuntimeError::InvalidNumberOfParameters);
    }

    for i in 0..arity {
        ctx.frames.declare_variable(&params[i], args[i].clone());
    }

    return interpret_stmt(ctx, body);
}

fn binary_expr(
    ctx: &mut Context,
    lhs: &Box<Expr>,
    rhs: &Box<Expr>,
    op: &Token,
) -> Result<Value, RuntimeError> {
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

fn add(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs + rhs)),
        _ => todo!(),
    }
}

fn sub(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs - rhs)),
        _ => todo!(),
    }
}

fn mul(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs * rhs)),
        _ => todo!(),
    }
}

fn div(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs / rhs)),
        _ => todo!(),
    }
}

fn and(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;

    if let Value::Boolean(lhs) = lhs {
        if !lhs {
            return Ok(Value::Boolean(false));
        }

        let rhs = interpret_expr(ctx, rhs)?;

        if let Value::Boolean(rhs) = rhs {
            return Ok(Value::Boolean(lhs && rhs));
        }
    }

    todo!();
}

fn or(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;

    if let Value::Boolean(lhs) = lhs {
        if lhs {
            return Ok(Value::Boolean(true));
        }

        let rhs = interpret_expr(ctx, rhs)?;

        if let Value::Boolean(rhs) = rhs {
            return Ok(Value::Boolean(lhs || rhs));
        }
    }

    todo!();
}

fn lt(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Boolean(lhs < rhs)),
        _ => unreachable!(),
    }
}

fn le(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Boolean(lhs <= rhs)),
        _ => unreachable!(),
    }
}

fn gt(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Boolean(lhs > rhs)),
        _ => unreachable!(),
    }
}

fn ge(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Boolean(lhs >= rhs)),
        _ => unreachable!(),
    }
}

fn eq(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    return Ok(Value::Boolean(lhs == rhs));
}

fn neq(ctx: &mut Context, lhs: &Box<Expr>, rhs: &Box<Expr>) -> Result<Value, RuntimeError> {
    let lhs = interpret_expr(ctx, lhs)?;
    let rhs = interpret_expr(ctx, rhs)?;

    return Ok(Value::Boolean(lhs != rhs));
}

fn unary_expr(ctx: &mut Context, right: &Box<Expr>, op: &Token) -> Result<Value, RuntimeError> {
    let right_value = interpret_expr(ctx, right)?;

    match op {
        Token::Minus => {
            if let Value::Number(v) = right_value {
                Ok(Value::Number(-v))
            } else {
                todo!();
            }
        }
        Token::Not => {
            if let Value::Boolean(b) = right_value {
                Ok(Value::Boolean(!b))
            } else {
                todo!();
            }
        }
        _ => todo!(),
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

        let writer = TestWriter::new();
        let mut ctx = Context::with_writer(Box::new(writer.clone()));

        interpret_with_context(&mut ctx, &program);

        let output = writer.get_output();

        assert_eq!(expected, output);
    }

    #[test]
    fn print_call() {
        test_program_output("print(42);", "42\n");
        test_program_output("print(42, 43, 44);", "42 43 44\n");
    }

    #[test]
    fn variables() {
        test_program_output(
            r#"
        var a = 42;
        var b = 1337;

        print(a + b);
        "#,
            "1379\n",
        );

        test_program_output(
            r#"
            var a = 42;
            {
                var a = 43;
                print(a);
            }
            print(a);
            "#,
            "43\n42\n",
        );

        test_program_output(
            r#"
        var a = 42;
        a = 43;
        print(a);
        "#,
            "43\n",
        );
    }

    #[test]
    fn if_stmt() {
        test_program_output(
            r#"
            var a = 1;
            if a == 1 {
                print(a);
            }

            if a == 2 {
                print(a, a);
            }
            "#,
            "1\n",
        );

        test_program_output(
            r#"
            var a = 1;
            if a == 0 {
            } else { 
                print(a);
            }"#,
            "1\n",
        );

        test_program_output(
            r#"
            var a = 1;
            if a == 0 {
            } else if a == 1{ 
                print(a);
            }"#,
            "1\n",
        );

        test_program_output(
            r#"
            var a = 2;
            if a == 0 {
            } else if a == 1 {
            } else if a == 2 {
                print(a);
            }"#,
            "2\n",
        );
    }

    #[test]
    fn while_stmt() {
        test_program_output(
            r#"
            var a = 0;
            while a < 5 {
                print(a);
                a += 1;
            }"#,
            "0\n1\n2\n3\n4\n",
        );

        test_program_output(
            r#"
            var a = 0;
            while a < 10 {
                if a == 5 {
                    return;
                }

                print(a);
                a += 1;
            }"#,
            "0\n1\n2\n3\n4\n",
        );
    }

    #[test]
    fn functions() {
        test_program_output(
            r#"
            fn foo(a) {
                print(a);
            }

            foo(42);"#,
            "42\n",
        );

        test_program_output(
            r#"
            fn foo(a) { 
                return a * 2;
            }

            print(foo(21));"#,
            "42\n",
        );

        test_program_output(
            r#"
            fn fib(n) {
                if n == 0 { return 0; }
                if n == 1 { return 1; }

                return fib(n-1) + fib(n - 2);
            }

            print(fib(23));
            "#,
            "28657\n",
        );
    }
}

#[cfg(test)]
mod expression_tests {
    use crate::{lexer::tokenize, parser::parse};

    use super::*;

    fn test_eval_expression(input: &str, expected: Result<Value, RuntimeError>) {
        let (tokens, errors) = tokenize(input);
        assert_eq!(errors, vec![]);

        let program = parse(tokens);
        assert_eq!(program.errors, vec![]);
        assert_eq!(program.statements.len(), 1);

        let mut ctx = Context::new();

        match *program.statements[0] {
            Stmt::Expr(ref expr) => {
                let result = interpret_expr(&mut ctx, expr);

                assert_eq!(result, expected);
            }
            _ => assert!(false),
        }
    }

    #[test]
    fn number() {
        test_eval_expression("2;", Ok(Value::Number(2.0)));
    }

    #[test]
    fn boolean() {
        test_eval_expression("true;", Ok(Value::Boolean(true)));
        test_eval_expression("false;", Ok(Value::Boolean(false)));
    }

    #[test]
    fn unary() {
        test_eval_expression("-42.0;", Ok(Value::Number(-42.0)));
        test_eval_expression("--42.0;", Ok(Value::Number(42.0)));
        test_eval_expression("not true;", Ok(Value::Boolean(false)));
        test_eval_expression("not not not not false;", Ok(Value::Boolean(false)));
    }

    #[test]
    fn binary_arithmetic() {
        test_eval_expression("3 + 2;", Ok(Value::Number(5.0)));
        test_eval_expression("4 - 1;", Ok(Value::Number(3.0)));
        test_eval_expression("10.5 * 4;", Ok(Value::Number(42.0)));
        test_eval_expression("336 / 8;", Ok(Value::Number(42.0)));

        test_eval_expression("3 + 5 * 4 - 12 / 2;", Ok(Value::Number(17.0)));

        test_eval_expression("2 * 3 + 4;", Ok(Value::Number(10.0)));
        test_eval_expression("2 * (3 + 4);", Ok(Value::Number(14.0)));
    }

    #[test]
    fn binary_logic() {
        test_eval_expression("true and false;", Ok(Value::Boolean(false)));
        test_eval_expression("false and 42.0;", Ok(Value::Boolean(false)));

        test_eval_expression("true or 1337;", Ok(Value::Boolean(true)));
        test_eval_expression("false or true;", Ok(Value::Boolean(true)));
    }

    #[test]
    fn comparisons() {
        test_eval_expression("1 < 2;", Ok(Value::Boolean(true)));
        test_eval_expression("2 <= 1;", Ok(Value::Boolean(false)));
        test_eval_expression("4 > 3.99;", Ok(Value::Boolean(true)));
        test_eval_expression("3.99 >= 4.01;", Ok(Value::Boolean(false)));
        test_eval_expression("true == false;", Ok(Value::Boolean(false)));
        test_eval_expression("true != false;", Ok(Value::Boolean(true)));
    }

    #[test]
    fn identifier_undeclared() {
        test_eval_expression(
            "a;",
            Err(RuntimeError::VariableNotDeclared("a".to_string())),
        );

        test_eval_expression(
            "print(42);",
            Err(RuntimeError::FunctionNotDeclared("print".to_string())),
        );
    }
}
