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

use crate::lexer::Token;

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr(Box<Expr>),
}

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
    Call {
        callee: Box<Expr>,
        arguments: Vec<Box<Expr>>,
    },
    Assignment {
        identifier: String,
        value: Box<Expr>,
    },
    Number(f64),
    Boolean(bool),
    // TODO: Proper string memory management
    String(String),
    Identifier(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken {
        expected: Token,
        found: Token,
        // TODO: Replace with line info
        position: usize,
    },
    InvalidTokenType {
        found: Token,
        // TODO: Replace with line info
        position: usize,
    },
}

pub struct Program {
    exprs: Vec<Expr>,
    errors: Vec<ParseError>,
}

// NOTE: Will change
pub fn parse(tokens: Vec<Token>) -> Result<Box<Expr>, ParseError> {
    let mut parser = Parser::new(tokens);
    return parser.parse_program();
}

struct Parser {
    tokens: Vec<Token>,
    current_index: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current_index: 0,
        }
    }

    // NOTE: Will change
    fn parse_program(&mut self) -> Result<Box<Expr>, ParseError> {
        return self.expression();
    }

    // expression   -> assignment ;
    fn expression(&mut self) -> Result<Box<Expr>, ParseError> {
        let expr = self.assignement()?;

        return Ok(expr);
    }

    // assignment   -> IDENTIFIER "=" assignment | logic_or ;
    fn assignement(&mut self) -> Result<Box<Expr>, ParseError> {
        let current_index = self.current_index;
        let expr = self.logic_or()?;

        if self.matches(Token::Equal) {
            let value = self.logic_or()?;

            match *expr {
                Expr::Identifier(identifier) => {
                    return Ok(Box::new(Expr::Assignment { identifier, value }));
                }
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: Token::Identifier("".to_string()),
                        found: self.tokens[current_index].clone(),
                        position: current_index,
                    });
                }
            }
        }

        return Ok(expr);
    }

    // logic_or     -> logic_and ( "or" logic_and )* ;
    fn logic_or(&mut self) -> Result<Box<Expr>, ParseError> {
        let mut expr = self.logic_and()?;

        while self.matches(Token::Or) {
            let op = self.previous_token().clone();
            let rhs = self.logic_and()?;
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return Ok(expr);
    }

    // logic_and    -> equality ( "and" equality )* ;
    fn logic_and(&mut self) -> Result<Box<Expr>, ParseError> {
        let mut expr = self.equality()?;

        while self.matches(Token::And) {
            let op = self.previous_token().clone();
            let rhs = self.logic_and()?;
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return Ok(expr);
    }

    // equality     -> comparison ( ( "!=" | "==" ) comparison )* ;
    fn equality(&mut self) -> Result<Box<Expr>, ParseError> {
        let mut expr = self.comparison()?;

        while self.matches_any(&[Token::BangEqual, Token::EqualEqual]) {
            let op = self.previous_token().clone();
            let rhs = self.factor()?;
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return Ok(expr);
    }

    // comparison   -> term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
    fn comparison(&mut self) -> Result<Box<Expr>, ParseError> {
        let mut expr = self.term()?;

        while self.matches_any(&[
            Token::Greater,
            Token::GreaterEqual,
            Token::Less,
            Token::LessEqual,
        ]) {
            let op = self.previous_token().clone();
            let rhs = self.factor()?;
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return Ok(expr);
    }

    // term         -> factor ( ( "-" | "+" ) factor )* ;
    fn term(&mut self) -> Result<Box<Expr>, ParseError> {
        let mut expr = self.factor()?;

        while self.matches_any(&[Token::Minus, Token::Plus]) {
            let op = self.previous_token().clone();
            let rhs = self.factor()?;
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return Ok(expr);
    }

    // factor       -> unary ( ( "/" | "*" ) unary )* ;
    fn factor(&mut self) -> Result<Box<Expr>, ParseError> {
        let mut expr = self.unary()?;

        while self.matches_any(&[Token::Asterisk, Token::Slash]) {
            let op = self.previous_token().clone();
            let rhs = self.unary()?;
            let lhs = expr;

            expr = Box::new(Expr::Binary { lhs, rhs, op });
        }

        return Ok(expr);
    }

    // unary        -> ("not" | "-") unary
    //               | call ;
    fn unary(&mut self) -> Result<Box<Expr>, ParseError> {
        if self.matches_any(&[Token::Minus, Token::Not]) {
            let op = self.previous_token().clone();
            let rhs = self.unary()?;

            return Ok(Box::new(Expr::Unary { rhs, op }));
        }

        return self.call();
    }

    // call         -> primary ( "(" arguments? ")" "* ;
    // arguments    -> expression ( "," expression )* ","? ;
    fn call(&mut self) -> Result<Box<Expr>, ParseError> {
        let start_index = self.current_index;
        let mut expr = self.primary()?;

        loop {
            if self.matches(Token::OpenParen) {
                // FIXME: Should an error be raised if the expr is not an identifier
                // in that case ?

                let callee = expr;
                expr = self.finish_call(callee)?;
            } else {
                break;
            }
        }

        return Ok(expr);
    }

    fn finish_call(&mut self, callee: Box<Expr>) -> Result<Box<Expr>, ParseError> {
        let mut arguments = vec![];

        if !self.check(Token::CloseParen) {
            loop {
                arguments.push(self.expression()?);

                if !self.matches(Token::Comma) {
                    break;
                }
            }
        }

        // TODO: Allow for a leading comma, this is currently not the case

        _ = self.expect(Token::CloseParen)?;

        return Ok(Box::new(Expr::Call { callee, arguments }));
    }

    // primary      -> NUMBER | STRING | "true" | "false" | "nil"
    //               | "(" expression ")" | IDENTIFIER;
    fn primary(&mut self) -> Result<Box<Expr>, ParseError> {
        let expr = match self.current_token() {
            Token::Number(n) => Box::new(Expr::Number(*n)),
            Token::True => Box::new(Expr::Boolean(true)),
            Token::False => Box::new(Expr::Boolean(false)),
            Token::String(s) => Box::new(Expr::String(s.clone())),
            Token::Identifier(s) => Box::new(Expr::Identifier(s.clone())),
            _ => {
                return Err(ParseError::InvalidTokenType {
                    found: self.current_token().clone(),
                    position: self.current_index,
                });
            }
        };

        self.advance();

        return Ok(expr);
    }

    fn matches(&mut self, token: Token) -> bool {
        if variant_eq(self.current_token(), &token) {
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
        return variant_eq(self.current_token(), &token);
    }

    fn expect(&mut self, expected: Token) -> Result<&Token, ParseError> {
        if variant_eq(self.current_token(), &expected) {
            return Ok(self.advance());
        }

        return Err(ParseError::UnexpectedToken {
            expected: expected.clone(),
            found: self.current_token().clone(),
            position: self.current_index,
        });
    }

    fn current_token(&self) -> &Token {
        return &self.tokens[self.current_index];
    }

    fn previous_token(&self) -> &Token {
        return &self.tokens[self.current_index - 1];
    }

    fn advance(&mut self) -> &Token {
        if self.current_index + 1 < self.tokens.len() {
            self.current_index += 1;
            return self.previous_token();
        }

        return self.current_token();
    }
}

fn variant_eq<T>(a: &T, b: &T) -> bool {
    return std::mem::discriminant(a) == std::mem::discriminant(b);
}

#[cfg(test)]
mod tests {
    use crate::lexer::tokenize;

    use super::*;

    fn test_expression(input: &str, expected: Result<Box<Expr>, ParseError>) {
        let (tokens, errors) = tokenize(input);
        assert_eq!(errors, vec![]);

        let mut parser = Parser::new(tokens);

        let expr = parser.expression();

        assert_eq!(expr, expected);
    }

    #[test]
    fn primary_expressions() {
        test_expression("42", Ok(Box::new(Expr::Number(42.0))));
        test_expression("true", Ok(Box::new(Expr::Boolean(true))));
        test_expression(
            "\"hello, world\"",
            Ok(Box::new(Expr::String("hello, world".into()))),
        );

        test_expression(
            "if",
            Err(ParseError::InvalidTokenType {
                found: Token::If,
                position: 0,
            }),
        );
    }

    #[test]
    fn unary_expressions() {
        test_expression(
            "-42",
            Ok(Box::new(Expr::Unary {
                rhs: Box::new(Expr::Number(42.0)),
                op: Token::Minus,
            })),
        );

        test_expression(
            "not false",
            Ok(Box::new(Expr::Unary {
                rhs: Box::new(Expr::Boolean(false)),
                op: Token::Not,
            })),
        );
    }

    #[test]
    fn term_expressions() {
        test_expression(
            "1 + 2",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(1.0)),
                rhs: Box::new(Expr::Number(2.0)),
                op: Token::Plus,
            })),
        );

        test_expression(
            "4.2 - 13.37",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(4.2)),
                rhs: Box::new(Expr::Number(13.37)),
                op: Token::Minus,
            })),
        );
    }

    #[test]
    fn factor_expressions() {
        test_expression(
            "2 * 3",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(2.0)),
                rhs: Box::new(Expr::Number(3.0)),
                op: Token::Asterisk,
            })),
        );

        test_expression(
            "7 / 4",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(7.0)),
                rhs: Box::new(Expr::Number(4.0)),
                op: Token::Slash,
            })),
        );
    }

    #[test]
    fn logic_or_expressions() {
        test_expression(
            "true or false",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Boolean(true)),
                rhs: Box::new(Expr::Boolean(false)),
                op: Token::Or,
            })),
        );
    }

    #[test]
    fn logic_and_expressions() {
        test_expression(
            "false and true",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Boolean(false)),
                rhs: Box::new(Expr::Boolean(true)),
                op: Token::And,
            })),
        );
    }

    #[test]
    fn equality_expressions() {
        test_expression(
            "33.0 == false",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(33.0)),
                rhs: Box::new(Expr::Boolean(false)),
                op: Token::EqualEqual,
            })),
        );

        test_expression(
            "\"hello\" != \"test\"",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::String("hello".into())),
                rhs: Box::new(Expr::String("test".into())),
                op: Token::BangEqual,
            })),
        );
    }

    #[test]
    fn comparison_expressions() {
        test_expression(
            "42 > 33",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::Greater,
            })),
        );

        test_expression(
            "42 >= 33",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::GreaterEqual,
            })),
        );

        test_expression(
            "42 < 33",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::Less,
            })),
        );

        test_expression(
            "42 <= 33",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(42.0)),
                rhs: Box::new(Expr::Number(33.0)),
                op: Token::LessEqual,
            })),
        );
    }

    #[test]
    fn call_expressions() {
        test_expression(
            "foo()",
            Ok(Box::new(Expr::Call {
                callee: Box::new(Expr::Identifier("foo".to_string())),
                arguments: vec![],
            })),
        );

        test_expression(
            "bar(\"test\", 39 + 3)",
            Ok(Box::new(Expr::Call {
                callee: Box::new(Expr::Identifier("bar".to_string())),
                arguments: vec![
                    Box::new(Expr::String("test".to_string())),
                    Box::new(Expr::Binary {
                        lhs: Box::new(Expr::Number(39.0)),
                        rhs: Box::new(Expr::Number(3.0)),
                        op: Token::Plus,
                    }),
                ],
            })),
        );

        // FIXME: Should this be syntaxically valid ?
        test_expression(
            "4()",
            Ok(Box::new(Expr::Call {
                callee: Box::new(Expr::Number(4.0)),
                arguments: vec![],
            })),
        );

        test_expression(
            "baz(42",
            Err(ParseError::UnexpectedToken {
                expected: Token::CloseParen,
                found: Token::EOF,
                position: 3,
            }),
        );
    }

    #[test]
    fn assignment_expressions() {
        test_expression(
            "a = 42",
            Ok(Box::new(Expr::Assignment {
                identifier: "a".to_string(),
                value: Box::new(Expr::Number(42.0)),
            })),
        );

        test_expression(
            "foo=15+27",
            Ok(Box::new(Expr::Assignment {
                identifier: "foo".to_string(),
                value: Box::new(Expr::Binary {
                    lhs: Box::new(Expr::Number(15.0)),
                    rhs: Box::new(Expr::Number(27.0)),
                    op: Token::Plus,
                }),
            })),
        );

        test_expression(
            "27 = 42",
            Err(ParseError::UnexpectedToken {
                expected: Token::Identifier("".to_string()),
                found: Token::Number(27.0),
                position: 0,
            }),
        );
    }
}
