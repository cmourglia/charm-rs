// program          -> declaration* EOF ;
// declaration      -> var_decl | fun_decl | statement
// statement        -> expr_stmt | print_stmt | if_stmt | block_stmt
//                   | while_stmt | for_stmt | return_stmt ;
// expr_stmt        -> expression ";" ;
// var_decl         -> "var" IDENTIFIER ( "=" expression )? ";" ;
// function_decl    -> "function" function ;
// function         -> IDENTIFIER "(" parameters? ")" block_stmt ;
// if_stmt          -> "if" expression block_stmt
//                     ( "else" if_stmt | block_stmt ) ? ;
// block_stmt       -> "{" ( statement )* "}" ;
// while_stmt       -> "while" expression block_stmt ;
// for_stmt         -> "for" ( var_decl | expr_stmt | ";" )
//                     expression? ";"
//                     expression? block_stmt ;
// return_stmt      -> "return" expression? ";" ;

// expression   -> assignment ;
// assignment   -> IDENTIFIER "=" assignment | logic_or ;
// logic_or     -> logic_and ( "or" logic_and )* ;
// logic_and    -> equality ( "and" equality )* ;
// equality     -> comparison ( ( "!=" | "==" ) comparison )* ;
// comparison   -> term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
// term         -> factor ( ( "-" | "+" ) factor )* ;
// factor       -> unary ( ( "/" | "*" ) unary )* ;
// unary        -> ("not" | "-") unary
//               | call ;
// call         -> primary ( "(" arguments? ")" "* ;
// arguments    -> expression ( "," expression )* ;
// primary      -> NUMBER | STRING | "true" | "false" | "nil"
//               | "(" expression ")" | IDENTIFIER;

use crate::lexer::{Lexer, Token};

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Binary {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        op: Token,
    },
    Unary {
        rhs: Box<Expr>,
        op: Token,
    },
    Number(f64),
    Boolean(bool),
    // TODO: Proper string memory management
    String(String),
}

pub struct Program {
    exprs: Vec<Expr>,
}

pub fn parse_program(input: &str) -> Box<Expr> {
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer);

    return parser.parse_program();
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,

    current_token: Token,
    prev_token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        Self {
            lexer,
            current_token: Token::Invalid(String::from("Not-initialized")),
            prev_token: Token::Invalid(String::from("Not-initialized")),
        }
    }

    // NOTE: Will change
    pub fn parse_program(&mut self) -> Box<Expr> {
        self.advance();
        return self.expression();
    }

    // expression   -> assignment ;
    fn expression(&mut self) -> Box<Expr> {
        return self.logic_or();
        // return self.assignement();
    }

    // assignment   -> IDENTIFIER "=" assignment | logic_or ;
    fn assignement(&mut self) -> Box<Expr> {
        let expr = self.logic_or();

        return expr;
    }

    // logic_or     -> logic_and ( "or" logic_and )* ;
    fn logic_or(&mut self) -> Box<Expr> {
        let mut expr = self.logic_and();

        while self.matches(Token::Or) {
            let op = self.prev_token.clone();
            let rhs = self.logic_and();
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return expr;
    }

    // logic_and    -> equality ( "and" equality )* ;
    fn logic_and(&mut self) -> Box<Expr> {
        let mut expr = self.equality();

        while self.matches(Token::And) {
            let op = self.prev_token.clone();
            let rhs = self.logic_and();
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return expr;
    }

    // equality     -> comparison ( ( "!=" | "==" ) comparison )* ;
    fn equality(&mut self) -> Box<Expr> {
        let mut expr = self.comparison();

        while self.matches_any(&[Token::BangEqual, Token::EqualEqual]) {
            let op = self.prev_token.clone();
            let rhs = self.factor();
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return expr;
    }

    // comparison   -> term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
    fn comparison(&mut self) -> Box<Expr> {
        let mut expr = self.term();

        while self.matches_any(&[
            Token::Greater,
            Token::GreaterEqual,
            Token::Less,
            Token::LessEqual,
        ]) {
            let op = self.prev_token.clone();
            let rhs = self.factor();
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return expr;
    }

    // term         -> factor ( ( "-" | "+" ) factor )* ;
    fn term(&mut self) -> Box<Expr> {
        let mut expr = self.factor();

        while self.matches_any(&[Token::Minus, Token::Plus]) {
            let op = self.prev_token.clone();
            let rhs = self.factor();
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return expr;
    }

    // factor       -> unary ( ( "/" | "*" ) unary )* ;
    fn factor(&mut self) -> Box<Expr> {
        let mut expr = self.unary();

        while self.matches_any(&[Token::Asterisk, Token::Slash]) {
            let op = self.prev_token.clone();
            let rhs = self.unary();
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return expr;
    }

    // unary        -> ("not" | "-") unary
    //               | call ;
    fn unary(&mut self) -> Box<Expr> {
        if self.matches_any(&[Token::Minus, Token::Not]) {
            let op = self.prev_token.clone();
            let rhs = self.unary();

            return Box::new(Expr::Unary { rhs, op });
        }

        return self.primary();
    }

    // call         -> primary ( "(" arguments? ")" "* ;
    fn call(&mut self) -> Box<Expr> {
        todo!();
    }

    // arguments    -> expression ( "," expression )* ;
    fn arguments(&mut self) -> Box<Expr> {
        todo!();
    }

    // primary      -> NUMBER | STRING | "true" | "false" | "nil"
    //               | "(" expression ")" | IDENTIFIER;
    fn primary(&mut self) -> Box<Expr> {
        let expr = match self.current_token {
            Token::Number(n) => Box::new(Expr::Number(n)),
            Token::True => Box::new(Expr::Boolean(true)),
            Token::False => Box::new(Expr::Boolean(false)),
            Token::String(ref s) => Box::new(Expr::String(s.clone())),
            _ => unreachable!("Invalid token type: {:?}", &self.current_token),
        };

        self.advance();

        return expr;
    }

    fn matches(&mut self, token: Token) -> bool {
        if variant_eq(&self.current_token, &token) {
            self.advance();
            return true;
        }

        return false;
    }

    fn matches_any(&mut self, tokens: &[Token]) -> bool {
        for token in tokens {
            if self.matches(token.clone()) {
                return true;
            }
        }

        return false;
    }

    fn check(&self, token: Token) -> bool {
        return variant_eq(&self.current_token, &token);
    }

    fn expect(&mut self, expected: Token) -> Token {
        if variant_eq(&self.current_token, &expected) {
            return self.advance();
        }

        println!(
            "Expected token `{:?}`, found `{:?}`",
            &expected, &self.current_token
        );

        unreachable!();
    }

    fn advance(&mut self) -> Token {
        match self.lexer.next() {
            Some(token) => {
                self.prev_token = self.current_token.clone();
                self.current_token = token;
            }

            None => {}
        }

        return self.prev_token.clone();
    }
}

fn variant_eq<T>(a: &T, b: &T) -> bool {
    return std::mem::discriminant(a) == std::mem::discriminant(b);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_expression(input: &str, expected: Box<Expr>) {
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);

        parser.advance();
        let expr = parser.expression();

        assert_eq!(expr, expected);
    }

    #[test]
    fn primary_expressions() {
        test_expression("42", Box::new(Expr::Number(42.0)));
        test_expression("true", Box::new(Expr::Boolean(true)));
        test_expression(
            "\"hello, world\"",
            Box::new(Expr::String("hello, world".into())),
        );
    }

    #[test]
    fn unary_expressions() {
        test_expression(
            "-42",
            Box::new(Expr::Unary {
                rhs: Box::new(Expr::Number(42.0)),
                op: Token::Minus,
            }),
        );

        test_expression(
            "not false",
            Box::new(Expr::Unary {
                rhs: Box::new(Expr::Boolean(false)),
                op: Token::Not,
            }),
        );
    }

    #[test]
    fn term_expressions() {
        test_expression(
            "1 + 2",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(1.0)),
                rhs: Box::new(Expr::Number(2.0)),
                op: Token::Plus,
            }),
        );

        test_expression(
            "4.2 - 13.37",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(4.2)),
                rhs: Box::new(Expr::Number(13.37)),
                op: Token::Minus,
            }),
        );
    }

    #[test]
    fn factor_expressions() {
        test_expression(
            "2 * 3",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(2.0)),
                rhs: Box::new(Expr::Number(3.0)),
                op: Token::Asterisk,
            }),
        );

        test_expression(
            "7 / 4",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(7.0)),
                rhs: Box::new(Expr::Number(4.0)),
                op: Token::Slash,
            }),
        );
    }

    #[test]
    fn logic_or_expressions() {
        test_expression(
            "true or false",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Boolean(true)),
                rhs: Box::new(Expr::Boolean(false)),
                op: Token::Or,
            }),
        );
    }

    #[test]
    fn logic_and_expressions() {
        test_expression(
            "false and true",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Boolean(false)),
                rhs: Box::new(Expr::Boolean(true)),
                op: Token::And,
            }),
        );
    }

    #[test]
    fn equality_expressions() {
        test_expression(
            "33.0 == false",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(33.0)),
                rhs: Box::new(Expr::Boolean(false)),
                op: Token::EqualEqual,
            }),
        );

        test_expression(
            "\"hello\" != \"test\"",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::String("hello".into())),
                rhs: Box::new(Expr::String("test".into())),
                op: Token::BangEqual,
            }),
        );
    }

    #[test]
    fn comparison_expressions() {
        test_expression(
            "42 > 33",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::Greater,
            }),
        );

        test_expression(
            "42 >= 33",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::GreaterEqual,
            }),
        );

        test_expression(
            "42 < 33",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::Less,
            }),
        );

        test_expression(
            "42 <= 33",
            Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::LessEqual,
            }),
        );
    }
}
