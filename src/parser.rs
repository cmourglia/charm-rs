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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add,
    Substract,
    Multiply,
    Divide,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Binary {
        left: Box<Expr>,
        right: Box<Expr>,
        op: BinaryOp,
    },
    Number(f32),
}

struct Program {
    exprs: Vec<Expr>,
}

pub struct Parser<'a> {
    lexer: &'a mut Lexer<'a>,

    current_token: Token<'a>,
    prev_token: Token<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: &'a mut Lexer<'a>) -> Self {
        Self {
            lexer,
            current_token: Token::Invalid("Not-initialized"),
            prev_token: Token::Invalid("Not-initialized"),
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
            let op = match self.prev_token {
                Token::Minus => BinaryOp::Substract,
                Token::Plus => BinaryOp::Add,
                _ => unreachable!(),
            };

            let right = self.factor();

            let left = expr;

            expr = Box::new(Expr::Binary { left, right, op });
        }

        return expr;
    }

    // factor       -> unary ( ( "/" | "*" ) unary )* ;
    fn factor(&mut self) -> Box<Expr> {
        // TODO: unary
        let mut expr = self.primary();

        while self.matches(Token::Asterisk) || self.matches(Token::Slash) {
            let op = match self.prev_token {
                Token::Asterisk => BinaryOp::Multiply,
                Token::Slash => BinaryOp::Divide,
                _ => unreachable!(),
            };

            // TODO: unary
            let right = self.primary();

            let left = expr;

            expr = Box::new(Expr::Binary { left, right, op });
        }

        return expr;
    }

    // unary        -> ("not" | "-") unary
    //               | call ;
    fn unary(&mut self) -> Box<Expr> {
        todo!();
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
        match self.current_token {
            Token::Number(str) => {
                self.advance();
                return Box::new(Expr::Number(str.parse::<f32>().unwrap()));
            }
            _ => unreachable!("Invalid token type"),
        }
    }

    fn matches(&mut self, token: Token<'a>) -> bool {
        if variant_eq(&self.current_token, &token) {
            self.advance();
            return true;
        }

        return false;
    }

    fn check(&self, token: Token<'a>) -> bool {
        return variant_eq(&self.current_token, &token);
    }

    fn expect(&mut self, expected: Token<'a>) -> Token<'a> {
        if variant_eq(&self.current_token, &expected) {
            return self.advance();
        }

        println!(
            "Expected token `{:?}`, found `{:?}`",
            &expected, &self.current_token
        );

        unreachable!();
    }

    fn advance(&mut self) -> Token<'a> {
        match self.lexer.next() {
            Some(token) => {
                self.prev_token = self.current_token;
                self.current_token = token;
            }

            None => {}
        }

        return self.prev_token;
    }
}

fn variant_eq<T>(a: &T, b: &T) -> bool {
    return std::mem::discriminant(a) == std::mem::discriminant(b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression() {
        // 42
        let mut lexer = Lexer::new("42");
        let mut parser = Parser::new(&mut lexer);

        parser.advance();
        let expr = parser.expression();

        assert_eq!(expr, Box::new(Expr::Number(42.0)));

        // 1 + 2
        let mut lexer = Lexer::new("1 + 2");
        let mut parser = Parser::new(&mut lexer);

        parser.advance();
        let expr = parser.expression();

        assert_eq!(
            expr,
            Box::new(Expr::Binary {
                left: Box::new(Expr::Number(1.0)),
                right: Box::new(Expr::Number(2.0)),
                op: BinaryOp::Add
            })
        );

        // 2 * 3
        let mut lexer = Lexer::new("2 * 3");
        let mut parser = Parser::new(&mut lexer);

        parser.advance();
        let expr = parser.expression();

        assert_eq!(
            expr,
            Box::new(Expr::Binary {
                left: Box::new(Expr::Number(2.0)),
                right: Box::new(Expr::Number(3.0)),
                op: BinaryOp::Multiply
            })
        );

        // 2 + 1 * 4 - 5 / 2
        let mut lexer = Lexer::new("2 + 1 * 4 - 5 / 2");
        let mut parser = Parser::new(&mut lexer);

        parser.advance();
        let expr = parser.expression();

        assert_eq!(
            expr,
            Box::new(Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(Expr::Number(2.0)),
                    right: Box::new(Expr::Binary {
                        left: Box::new(Expr::Number(1.0)),
                        right: Box::new(Expr::Number(4.0)),
                        op: BinaryOp::Multiply
                    }),
                    op: BinaryOp::Add
                }),
                right: Box::new(Expr::Binary {
                    left: Box::new(Expr::Number(5.0)),
                    right: Box::new(Expr::Number(2.0)),
                    op: BinaryOp::Divide
                }),
                op: BinaryOp::Substract
            })
        );
    }
}
