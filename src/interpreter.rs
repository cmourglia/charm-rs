use crate::lexer::Token;
use crate::parser::{Expr, Program, Stmt};
use crate::value::Value;

pub struct Context<W: std::io::Write = std::io::Stdout> {
    writer: W,
}

impl Context {
    pub fn new() -> Self {
        return Self::with_writer(std::io::stdout());
    }

    #[allow(unused)]
    pub fn with_writer<W: std::io::Write>(writer: W) -> Context<W> {
        return Context { writer };
    }
}

pub fn interpret(prg: &Program) {
    let mut ctx = Context::new();

    for stmt in &prg.statements {
        interpret_stmt(&mut ctx, stmt);
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
            callee: _,
            arguments: _,
        } => todo!(),
    }
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
mod tests {
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
