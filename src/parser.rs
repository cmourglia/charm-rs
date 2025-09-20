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
    VarDecl {
        identifier: String,
        expr: Option<Box<Expr>>,
    },
    FunctionDecl {
        identifier: String,
        // FIXME: This will need to be type declarations at some point instead
        args: Vec<String>,
        body: Box<Stmt>,
    },
    Block(Vec<Box<Stmt>>),
    If {
        cond: Box<Expr>,
        if_block: Box<Stmt>,
        else_block: Option<Box<Stmt>>,
    },
    While {
        cond: Box<Expr>,
        block: Box<Stmt>,
    },
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
        callee: String,
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
// TODO: Replace position with line info
pub enum ParseError {
    UnexpectedToken {
        expected: Token,
        found: Token,
        position: usize,
    },
    InvalidTokenType {
        found: Token,
        position: usize,
    },
    UnexpectedEndOfFile {
        expected: Token,
        position: usize,
    },
    Todo,
}

pub struct Program {
    pub statements: Vec<Box<Stmt>>,
    pub errors: Vec<ParseError>,
}

// NOTE: Will change
pub fn parse(tokens: Vec<Token>) -> Program {
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
    // program          -> declaration* EOF ;
    fn parse_program(&mut self) -> Program {
        let mut program = Program {
            statements: vec![],
            errors: vec![],
        };

        loop {
            let decl = self.declaration();

            match decl {
                Ok(stmt) => {
                    if let Some(stmt) = stmt {
                        program.statements.push(stmt);
                    } else {
                        break;
                    }
                }
                Err(err) => program.errors.push(err),
            };
        }

        return program;
    }

    // declaration      -> var_decl | fun_decl | statement
    fn declaration(&mut self) -> Result<Option<Box<Stmt>>, ParseError> {
        // TODO: Handle EOF ? Need to return an option ?

        match self.current_token() {
            Token::Var => Ok(Some(self.var_decl()?)),
            Token::Function => Ok(Some(self.function_decl()?)),
            Token::EOF => Ok(None),
            _ => Ok(Some(self.statement()?)),
        }
    }

    // statement        -> expr_stmt | if_stmt | block_stmt
    //                   | while_stmt | for_stmt | return_stmt ;
    fn statement(&mut self) -> Result<Box<Stmt>, ParseError> {
        match self.current_token() {
            Token::If => self.if_stmt(),
            Token::For => self.for_stmt(),
            Token::While => self.while_stmt(),
            Token::OpenBrace => self.block_stmt(),
            Token::Return => self.return_stmt(),
            _ => self.expr_stmt(),
        }
    }

    // expr_stmt        -> expression ";" ;
    fn expr_stmt(&mut self) -> Result<Box<Stmt>, ParseError> {
        let expr = self.expression()?;
        self.consume(Token::Semicolon)?;

        return Ok(Box::new(Stmt::Expr(expr)));
    }

    // var_decl         -> "var" IDENTIFIER ( "=" expression )? ";" ;
    fn var_decl(&mut self) -> Result<Box<Stmt>, ParseError> {
        self.consume(Token::Var)?;

        let identifier = self.identifier()?;

        let mut expr = None;

        if self.matches(Token::Equal) {
            expr = Some(self.expression()?);
        }

        self.consume(Token::Semicolon)?;

        return Ok(Box::new(Stmt::VarDecl { identifier, expr }));
    }

    // function_decl    -> "function" IDENTIFIER "(" parameters? ")" block_stmt ;
    fn function_decl(&mut self) -> Result<Box<Stmt>, ParseError> {
        self.consume(Token::Function)?;

        let identifier = self.identifier()?;

        let mut args = vec![];

        self.consume(Token::OpenParen)?;
        if !self.check(Token::CloseParen) {
            // TODO: Allow for trailing comma at some point
            loop {
                let arg = self.identifier()?;
                args.push(arg);

                if !self.matches(Token::Comma) {
                    break;
                }
            }
        }
        self.consume(Token::CloseParen)?;

        let body = self.block_stmt()?;

        return Ok(Box::new(Stmt::FunctionDecl {
            identifier,
            args,
            body,
        }));
    }

    // block_stmt       -> "{" ( statement )* "}" ;
    fn block_stmt(&mut self) -> Result<Box<Stmt>, ParseError> {
        let start_index = self.current_index;
        self.consume(Token::OpenBrace)?;

        let mut statements = vec![];

        loop {
            if self.matches(Token::CloseBrace) {
                break;
            }

            if let Some(stmt) = self.declaration()? {
                statements.push(stmt);
            } else {
                return Err(ParseError::UnexpectedEndOfFile {
                    expected: Token::CloseBrace,
                    position: start_index,
                });
            }
        }

        return Ok(Box::new(Stmt::Block(statements)));
    }

    // if_stmt          -> "if" expression block_stmt
    //                     ( "else" if_stmt | block_stmt ) ? ;
    fn if_stmt(&mut self) -> Result<Box<Stmt>, ParseError> {
        self.consume(Token::If)?;

        let cond = self.expression()?;

        let if_block = self.block_stmt()?;

        let mut else_block = None;

        if self.matches(Token::Else) {
            else_block = match self.current_token() {
                Token::If => Some(self.if_stmt()?),
                Token::OpenBrace => Some(self.block_stmt()?),
                _ => return Err(ParseError::Todo),
            };
        }

        return Ok(Box::new(Stmt::If {
            cond,
            if_block,
            else_block,
        }));
    }

    // while_stmt       -> "while" expression block_stmt ;
    fn while_stmt(&mut self) -> Result<Box<Stmt>, ParseError> {
        self.consume(Token::While)?;
        let cond = self.expression()?;
        let block = self.block_stmt()?;

        return Ok(Box::new(Stmt::While { cond, block }));
    }

    // for_stmt         -> "for" ( var_decl | expr_stmt | ";" )
    //                     expression? ";"
    //                     expression? block_stmt ;
    fn for_stmt(&mut self) -> Result<Box<Stmt>, ParseError> {
        todo!()
    }

    // return_stmt      -> "return" expression? ";" ;
    fn return_stmt(&mut self) -> Result<Box<Stmt>, ParseError> {
        todo!()
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
        let expr = self.primary()?;

        loop {
            if self.matches(Token::OpenParen) {
                let callee = expr;

                match *callee {
                    Expr::Identifier(identifier) => {
                        return Ok(self.finish_call(identifier)?);
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: Token::Identifier("".to_string()),
                            found: self.tokens[start_index].clone(),
                            position: start_index,
                        });
                    }
                };
            } else {
                break;
            }
        }

        return Ok(expr);
    }

    fn finish_call(&mut self, callee: String) -> Result<Box<Expr>, ParseError> {
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

        _ = self.consume(Token::CloseParen)?;

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
            Token::OpenParen => self.grouping()?,
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

    fn grouping(&mut self) -> Result<Box<Expr>, ParseError> {
        self.consume(Token::OpenParen)?;
        let expr = self.expression()?;
        self.expect(Token::CloseParen)?;
        return Ok(expr);
    }

    fn identifier(&mut self) -> Result<String, ParseError> {
        let token = self.consume(Token::Identifier(String::new()))?;

        match token {
            Token::Identifier(identifier) => Ok(identifier.clone()),
            _ => unreachable!("Already checked, should never ever get there"),
        }
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

    fn consume(&mut self, expected: Token) -> Result<&Token, ParseError> {
        if variant_eq(self.current_token(), &expected) {
            return Ok(self.advance());
        }

        return Err(ParseError::UnexpectedToken {
            expected: expected.clone(),
            found: self.current_token().clone(),
            position: self.current_index,
        });
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if !self.check(expected.clone()) {
            return Err(ParseError::UnexpectedToken {
                expected: expected.clone(),
                found: self.current_token().clone(),
                position: self.current_index,
            });
        }

        return Ok(());
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
mod stmt_tests {
    use crate::lexer::tokenize;

    use super::*;

    fn test_statement(input: &str, expected: Result<Option<Box<Stmt>>, ParseError>) {
        let (tokens, errors) = tokenize(input);
        assert_eq!(errors, vec![]);

        let mut parser = Parser::new(tokens);
        let stmt = parser.declaration();

        assert_eq!(stmt, expected);
    }

    #[test]
    fn empty_declaration() {
        test_statement("", Ok(None));
    }

    #[test]
    fn expr_stmt() {
        test_statement(
            "42;",
            Ok(Some(Box::new(Stmt::Expr(Box::new(Expr::Number(42.0)))))),
        );

        test_statement(
            "true or false and 42 < 43;",
            Ok(Some(Box::new(Stmt::Expr(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Boolean(true)),
                rhs: Box::new(Expr::Binary {
                    lhs: Box::new(Expr::Boolean(false)),
                    rhs: Box::new(Expr::Binary {
                        lhs: Box::new(Expr::Number(42.0)),
                        rhs: Box::new(Expr::Number(43.0)),
                        op: Token::Less,
                    }),
                    op: Token::And,
                }),
                op: Token::Or,
            }))))),
        );
    }

    #[test]
    fn var_decl() {
        test_statement(
            "var foo;",
            Ok(Some(Box::new(Stmt::VarDecl {
                identifier: "foo".to_string(),
                expr: None,
            }))),
        );

        test_statement(
            "var toto = \"tata\";",
            Ok(Some(Box::new(Stmt::VarDecl {
                identifier: "toto".to_string(),
                expr: Some(Box::new(Expr::String("tata".to_string()))),
            }))),
        );

        test_statement(
            "var bar",
            Err(ParseError::UnexpectedToken {
                expected: Token::Semicolon,
                found: Token::EOF,
                position: 2,
            }),
        );

        test_statement(
            "var baz = 42",
            Err(ParseError::UnexpectedToken {
                expected: Token::Semicolon,
                found: Token::EOF,
                position: 4,
            }),
        );
    }

    #[test]
    fn block_statement() {
        test_statement(
            r#"{
            }"#,
            Ok(Some(Box::new(Stmt::Block(vec![])))),
        );

        test_statement(
            r#"{
                42;
            }"#,
            Ok(Some(Box::new(Stmt::Block(vec![Box::new(Stmt::Expr(
                Box::new(Expr::Number(42.0)),
            ))])))),
        );

        test_statement(
            r#"{
                var foo = 42;
                var bar = 1337;
                var baz = foo + bar;
            }"#,
            Ok(Some(Box::new(Stmt::Block(vec![
                Box::new(Stmt::VarDecl {
                    identifier: "foo".to_string(),
                    expr: Some(Box::new(Expr::Number(42.0))),
                }),
                Box::new(Stmt::VarDecl {
                    identifier: "bar".to_string(),
                    expr: Some(Box::new(Expr::Number(1337.0))),
                }),
                Box::new(Stmt::VarDecl {
                    identifier: "baz".to_string(),
                    expr: Some(Box::new(Expr::Binary {
                        lhs: Box::new(Expr::Identifier("foo".to_string())),
                        rhs: Box::new(Expr::Identifier("bar".to_string())),
                        op: Token::Plus,
                    })),
                }),
            ])))),
        );

        test_statement(
            "{ 42; ",
            Err(ParseError::UnexpectedEndOfFile {
                expected: Token::CloseBrace,
                position: 0,
            }),
        );
    }

    #[test]
    fn function_decl() {
        test_statement(
            "function foo() {}",
            Ok(Some(Box::new(Stmt::FunctionDecl {
                identifier: "foo".to_string(),
                args: vec![],
                body: Box::new(Stmt::Block(vec![])),
            }))),
        );

        test_statement(
            "function foo(a) {}",
            Ok(Some(Box::new(Stmt::FunctionDecl {
                identifier: "foo".to_string(),
                args: vec!["a".to_string()],
                body: Box::new(Stmt::Block(vec![])),
            }))),
        );

        test_statement(
            "function foo(bar, baz) {}",
            Ok(Some(Box::new(Stmt::FunctionDecl {
                identifier: "foo".to_string(),
                args: vec!["bar".to_string(), "baz".to_string()],
                body: Box::new(Stmt::Block(vec![])),
            }))),
        );

        test_statement(
            "function foo(bar, baz {}",
            Err(ParseError::UnexpectedToken {
                expected: Token::CloseParen,
                found: Token::OpenBrace,
                position: 6,
            }),
        );
    }

    #[test]
    fn if_stmt() {
        test_statement(
            "if true {}",
            Ok(Some(Box::new(Stmt::If {
                cond: Box::new(Expr::Boolean(true)),
                if_block: Box::new(Stmt::Block(vec![])),
                else_block: None,
            }))),
        );

        test_statement(
            "if false {} else {}",
            Ok(Some(Box::new(Stmt::If {
                cond: Box::new(Expr::Boolean(false)),
                if_block: Box::new(Stmt::Block(vec![])),
                else_block: Some(Box::new(Stmt::Block(vec![]))),
            }))),
        );

        test_statement(
            "if false {} else if true {}",
            Ok(Some(Box::new(Stmt::If {
                cond: Box::new(Expr::Boolean(false)),
                if_block: Box::new(Stmt::Block(vec![])),
                else_block: Some(Box::new(Stmt::If {
                    cond: Box::new(Expr::Boolean(true)),
                    if_block: Box::new(Stmt::Block(vec![])),
                    else_block: None,
                })),
            }))),
        );

        test_statement(
            "if false {} else if true {} else if false {} else {}",
            Ok(Some(Box::new(Stmt::If {
                cond: Box::new(Expr::Boolean(false)),
                if_block: Box::new(Stmt::Block(vec![])),
                else_block: Some(Box::new(Stmt::If {
                    cond: Box::new(Expr::Boolean(true)),
                    if_block: Box::new(Stmt::Block(vec![])),
                    else_block: Some(Box::new(Stmt::If {
                        cond: Box::new(Expr::Boolean(false)),
                        if_block: Box::new(Stmt::Block(vec![])),
                        else_block: Some(Box::new(Stmt::Block(vec![]))),
                    })),
                })),
            }))),
        );

        test_statement(
            "if false else {}",
            Err(ParseError::UnexpectedToken {
                expected: Token::OpenBrace,
                found: Token::Else,
                position: 2,
            }),
        );

        test_statement("if false {} else foo;", Err(ParseError::Todo));
    }

    #[test]
    fn while_stmt() {
        test_statement(
            "while true {}",
            Ok(Some(Box::new(Stmt::While {
                cond: Box::new(Expr::Boolean(true)),
                block: Box::new(Stmt::Block(vec![])),
            }))),
        );

        test_statement(
            "while false;",
            Err(ParseError::UnexpectedToken {
                expected: Token::OpenBrace,
                found: Token::Semicolon,
                position: 2,
            }),
        );
    }

    // TODO: Test program
}

#[cfg(test)]
mod expr_tests {
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
    fn grouped_expressions() {
        test_expression("(42)", Ok(Box::new(Expr::Number(42.0))));
        test_expression(
            "1 * (2 + 3)",
            Ok(Box::new(Expr::Binary {
                lhs: Box::new(Expr::Number(1.0)),
                rhs: Box::new(Expr::Binary {
                    lhs: Box::new(Expr::Number(2.0)),
                    rhs: Box::new(Expr::Number(3.0)),
                    op: Token::Plus,
                }),
                op: Token::Asterisk,
            })),
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
                callee: "foo".to_string(),
                arguments: vec![],
            })),
        );

        test_expression(
            "bar(\"test\", 39 + 3)",
            Ok(Box::new(Expr::Call {
                callee: "bar".to_string(),
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

        test_expression(
            "4()",
            Err(ParseError::UnexpectedToken {
                expected: Token::Identifier("".to_string()),
                found: Token::Number(4.0),
                position: 0,
            }),
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
