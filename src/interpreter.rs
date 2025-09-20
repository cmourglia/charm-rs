use crate::lexer::Token;
use crate::parser::{Expr, Program, Stmt};
use crate::value::Value;

pub fn interpret(prg: &Program) {
    for stmt in &prg.statements {
        interpret_stmt(stmt);
    }
}

fn interpret_stmt(stmt: &Box<Stmt>) {
    todo!();
}

fn interpret_expr(expr: &Box<Expr>) -> Value {
    match **expr {
        Expr::Number(n) => Value::Number(n),
        Expr::Boolean(b) => Value::Boolean(b),

        Expr::Unary { ref rhs, ref op } => unary_expr(&rhs, op),

        Expr::Binary {
            ref lhs,
            ref rhs,
            ref op,
        } => binary_expr(&lhs, &rhs, op),

        _ => unreachable!(),
    }
}

fn binary_expr(lhs: &Box<Expr>, rhs: &Box<Expr>, op: &Token) -> Value {
    match op {
        Token::Plus => add(lhs, rhs),
        Token::Minus => sub(lhs, rhs),
        Token::Asterisk => mul(lhs, rhs),
        Token::Slash => div(lhs, rhs),
        Token::Greater => gt(lhs, rhs),
        Token::GreaterEqual => ge(lhs, rhs),
        Token::Less => lt(lhs, rhs),
        Token::LessEqual => le(lhs, rhs),
        Token::EqualEqual => eq(lhs, rhs),
        Token::BangEqual => neq(lhs, rhs),
        Token::Or => or(lhs, rhs),
        Token::And => and(lhs, rhs),
        _ => unreachable!(),
    }
}

fn add(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs + rhs),
        _ => unreachable!(),
    }
}

fn sub(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs - rhs),
        _ => unreachable!(),
    }
}

fn mul(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs * rhs),
        _ => unreachable!(),
    }
}

fn div(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Number(lhs / rhs),
        _ => unreachable!(),
    }
}

fn and(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);

    if let Value::Boolean(lhs) = lhs {
        if !lhs {
            return Value::Boolean(false);
        }

        let rhs = interpret_expr(rhs);

        if let Value::Boolean(rhs) = rhs {
            Value::Boolean(lhs && rhs)
        } else {
            unreachable!()
        }
    } else {
        unreachable!();
    }
}

fn or(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);

    if let Value::Boolean(lhs) = lhs {
        if lhs {
            return Value::Boolean(true);
        }

        let rhs = interpret_expr(rhs);

        if let Value::Boolean(rhs) = rhs {
            Value::Boolean(lhs || rhs)
        } else {
            unreachable!()
        }
    } else {
        unreachable!();
    }
}

fn lt(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs < rhs),
        _ => unreachable!(),
    }
}

fn le(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs <= rhs),
        _ => unreachable!(),
    }
}

fn gt(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs > rhs),
        _ => unreachable!(),
    }
}

fn ge(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    match (lhs, rhs) {
        (Value::Number(lhs), Value::Number(rhs)) => Value::Boolean(lhs >= rhs),
        _ => unreachable!(),
    }
}

fn eq(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    return Value::Boolean(lhs == rhs);
}

fn neq(lhs: &Box<Expr>, rhs: &Box<Expr>) -> Value {
    let lhs = interpret_expr(lhs);
    let rhs = interpret_expr(rhs);

    return Value::Boolean(lhs != rhs);
}

fn unary_expr(right: &Box<Expr>, op: &Token) -> Value {
    let right_value = interpret_expr(right);

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

        match *program.statements[0] {
            Stmt::Expr(ref expr) => {
                let value = interpret_expr(expr);
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
