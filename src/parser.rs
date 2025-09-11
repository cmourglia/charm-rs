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
        left: Box<Expr>,
        right: Box<Expr>,
        op: Token,
    },
    Unary {
        right: Box<Expr>,
        op: Token,
    },
    Number(f64),
    Boolean(bool),
    // TODO: Proper string memory management
    String(String),
}

struct Program {
    exprs: Vec<Expr>,
}

pub struct Parser<'a> {
    lexer: &'a mut Lexer<'a>,

    current_token: Token,
    prev_token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: &'a mut Lexer<'a>) -> Self {
        Self {
            lexer,
            current_token: Token::Invalid(String::from("Not-initialized")),
            prev_token: Token::Invalid(String::from("Not-initialized")),
        }
    }

    // NOTE: Will change
    pub fn parse_program(&mut self) -> Box<Expr> {
        return self.expression();
    }

    // expression   -> assignment ;
    fn expression(&mut self) -> Box<Expr> {
        return self.term();
        // return self.assignement();
    }

    // assignment   -> IDENTIFIER "=" assignment | logic_or ;
    fn assignement(&mut self) -> Box<Expr> {
        let expr = self.logic_or();

        return expr;
    }

    // logic_or     -> logic_and ( "or" logic_and )* ;
    fn logic_or(&mut self) -> Box<Expr> {
        let expr = self.logic_and();

        return expr;
    }

    // logic_and    -> equality ( "and" equality )* ;
    fn logic_and(&mut self) -> Box<Expr> {
        let expr = self.equality();

        return expr;
    }

    // equality     -> comparison ( ( "!=" | "==" ) comparison )* ;
    fn equality(&mut self) -> Box<Expr> {
        let expr = self.comparison();

        return expr;
    }

    // comparison   -> term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
    fn comparison(&mut self) -> Box<Expr> {
        let expr = self.term();

        return expr;
    }

    // term         -> factor ( ( "-" | "+" ) factor )* ;
    fn term(&mut self) -> Box<Expr> {
        let mut expr = self.factor();

        while self.matches(Token::Minus) || self.matches(Token::Plus) {
            let op = self.prev_token.clone();

            let right = self.factor();

            let left = expr;

            expr = Box::new(Expr::Binary { left, right, op });
        }

        return expr;
    }

    // factor       -> unary ( ( "/" | "*" ) unary )* ;
    fn factor(&mut self) -> Box<Expr> {
        let mut expr = self.unary();

        while self.matches(Token::Asterisk) || self.matches(Token::Slash) {
            let op = self.prev_token.clone();

            let right = self.unary();

            let left = expr;

            expr = Box::new(Expr::Binary { left, right, op });
        }

        return expr;
    }

    // unary        -> ("not" | "-") unary
    //               | call ;
    fn unary(&mut self) -> Box<Expr> {
        if self.matches(Token::Minus) || self.matches(Token::Not) {
            let op = self.prev_token.clone();

            let right = self.unary();

            return Box::new(Expr::Unary { right, op });
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
            _ => unreachable!("Invalid token type"),
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
        let mut lexer = Lexer::new(input);
        let mut parser = Parser::new(&mut lexer);

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
                right: Box::new(Expr::Number(42.0)),
                op: Token::Minus,
            }),
        );

        test_expression(
            "not false",
            Box::new(Expr::Unary {
                right: Box::new(Expr::Boolean(false)),
                op: Token::Not,
            }),
        );
    }

    #[test]
    fn binary_expressions() {
        test_expression(
            "1 + 2",
            Box::new(Expr::Binary {
                left: Box::new(Expr::Number(1.0)),
                right: Box::new(Expr::Number(2.0)),
                op: Token::Plus,
            }),
        );

        test_expression(
            "2 * 3",
            Box::new(Expr::Binary {
                left: Box::new(Expr::Number(2.0)),
                right: Box::new(Expr::Number(3.0)),
                op: Token::Asterisk,
            }),
        );

        test_expression(
            "2 + 1 * 4 - 5 / 2",
            Box::new(Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(Expr::Number(2.0)),
                    right: Box::new(Expr::Binary {
                        left: Box::new(Expr::Number(1.0)),
                        right: Box::new(Expr::Number(4.0)),
                        op: Token::Asterisk,
                    }),
                    op: Token::Plus,
                }),
                right: Box::new(Expr::Binary {
                    left: Box::new(Expr::Number(5.0)),
                    right: Box::new(Expr::Number(2.0)),
                    op: Token::Slash,
                }),
                op: Token::Minus,
            }),
        );
    }
}
