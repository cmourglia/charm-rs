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

use crate::{token::Token, utils::variant_eq};

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr(usize),
    VarDecl {
        identifier: String,
        expr: Option<usize>,
    },
    FunctionDecl {
        identifier: String,
        // FIXME: This will need to be type declarations at some point instead
        args: Vec<String>,
        body: usize,
    },
    Block(Vec<usize>),
    If {
        cond: usize,
        if_block: usize,
        else_block: Option<usize>,
    },
    While {
        cond: usize,
        block: usize,
    },
    Return(Option<usize>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Binary {
        lhs: usize,
        rhs: usize,
        op: Token,
    },
    Unary {
        rhs: usize,
        op: Token,
    },
    Call {
        callee: String,
        arguments: Vec<usize>,
    },
    Assignment {
        identifier: String,
        value: usize,
        op: Token,
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
    pub program_statements: Vec<usize>,
    pub errors: Vec<ParseError>,

    pub statements: Vec<Stmt>,
    pub expressions: Vec<Expr>,
}

// NOTE: Will change
pub fn parse(tokens: Vec<Token>) -> Program {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

struct Parser {
    tokens: Vec<Token>,
    current_index: usize,

    expressions: Vec<Expr>,
    statements: Vec<Stmt>,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current_index: 0,
            expressions: vec![],
            statements: vec![],
        }
    }

    // NOTE: Will change
    // program          -> declaration* EOF ;
    fn parse_program(&mut self) -> Program {
        let mut program = Program {
            program_statements: vec![],
            errors: vec![],
            statements: vec![],
            expressions: vec![],
        };

        loop {
            let decl = self.declaration();

            match decl {
                Ok(stmt) => {
                    if let Some(stmt) = stmt {
                        program.program_statements.push(stmt);
                    } else {
                        break;
                    }
                }
                Err(err) => program.errors.push(err),
            };
        }

        // TODO: Avoid this clone at some point
        program.expressions = self.expressions.clone();
        program.statements = self.statements.clone();

        program
    }

    // declaration      -> var_decl | fun_decl | statement
    fn declaration(&mut self) -> Result<Option<usize>, ParseError> {
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
    fn statement(&mut self) -> Result<usize, ParseError> {
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
    fn expr_stmt(&mut self) -> Result<usize, ParseError> {
        let expr = self.expression()?;
        self.consume(Token::Semicolon)?;

        Ok(self.push_statement(Stmt::Expr(expr)))
    }

    // var_decl         -> "var" IDENTIFIER ( "=" expression )? ";" ;
    fn var_decl(&mut self) -> Result<usize, ParseError> {
        self.consume(Token::Var)?;

        let identifier = self.identifier()?;

        let mut expr = None;

        if self.matches(Token::Equal) {
            expr = Some(self.expression()?);
        }

        self.consume(Token::Semicolon)?;

        Ok(self.push_statement(Stmt::VarDecl { identifier, expr }))
    }

    // function_decl    -> "function" IDENTIFIER "(" parameters? ")" block_stmt ;
    fn function_decl(&mut self) -> Result<usize, ParseError> {
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

        Ok(self.push_statement(Stmt::FunctionDecl {
            identifier,
            args,
            body,
        }))
    }

    // block_stmt       -> "{" ( statement )* "}" ;
    fn block_stmt(&mut self) -> Result<usize, ParseError> {
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

        Ok(self.push_statement(Stmt::Block(statements)))
    }

    // if_stmt          -> "if" expression block_stmt
    //                     ( "else" if_stmt | block_stmt ) ? ;
    fn if_stmt(&mut self) -> Result<usize, ParseError> {
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

        Ok(self.push_statement(Stmt::If {
            cond,
            if_block,
            else_block,
        }))
    }

    // while_stmt       -> "while" expression block_stmt ;
    fn while_stmt(&mut self) -> Result<usize, ParseError> {
        self.consume(Token::While)?;
        let cond = self.expression()?;
        let block = self.block_stmt()?;

        Ok(self.push_statement(Stmt::While { cond, block }))
    }

    // for_stmt         -> "for" (
    //                          ( var_decl | expr_stmt | ";" ) expression? ";" expression? )
    //              TODO:     | ( identifier "in" identifier )
    //                     ) block_stmt ;
    fn for_stmt(&mut self) -> Result<usize, ParseError> {
        // TODO: Handle `for i in v` at some point
        self.consume(Token::For)?;

        let init = match self.current_token() {
            Token::Var => Some(self.var_decl()?),
            Token::Semicolon => {
                self.advance();
                None
            }
            _ => Some(self.expr_stmt()?),
        };

        let cond = match self.current_token() {
            Token::Semicolon => self.push_expression(Expr::Boolean(true)),
            _ => self.expression()?,
        };
        self.consume(Token::Semicolon)?;

        let increment = match self.current_token() {
            Token::OpenBrace => None,
            _ => Some(self.expression()?),
        };

        let body = self.block_stmt()?;

        let mut body_statements = if let Stmt::Block(ref stmts) = self.statements[body] {
            stmts.clone()
        } else {
            // Does not make any sense to end up there
            unreachable!()
        };

        if let Some(increment) = increment {
            body_statements.push(self.push_statement(Stmt::Expr(increment)));
        }

        // Override the block statement at body
        self.statements[body] = Stmt::Block(body_statements);

        let while_stmt = self.push_statement(Stmt::While { cond, block: body });

        let mut statements = vec![];
        if let Some(init) = init {
            statements.push(init);
        }

        statements.push(while_stmt);
        let final_stmt = self.push_statement(Stmt::Block(statements));

        Ok(final_stmt)
    }

    // return_stmt      -> "return" expression? ";" ;
    fn return_stmt(&mut self) -> Result<usize, ParseError> {
        self.consume(Token::Return)?;

        let mut expr = None;

        if !self.check(Token::Semicolon) {
            expr = Some(self.expression()?);
        }

        self.consume(Token::Semicolon)?;

        Ok(self.push_statement(Stmt::Return(expr)))
    }

    // expression   -> assignment ;
    fn expression(&mut self) -> Result<usize, ParseError> {
        let expr = self.assignement()?;

        Ok(expr)
    }

    // assignment   -> IDENTIFIER "=" assignment | logic_or ;
    fn assignement(&mut self) -> Result<usize, ParseError> {
        let current_index = self.current_index;
        let expr = self.logic_or()?;

        let token = self.current_token().clone();

        if self.matches_any(&[
            Token::Equal,
            Token::PlusEqual,
            Token::MinusEqual,
            Token::AsteriskEqual,
            Token::SlashEqual,
        ]) {
            let value = self.logic_or()?;

            let identifier = match &self.expressions[expr] {
                Expr::Identifier(identifier) => identifier,
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: Token::Identifier("".to_string()),
                        found: self.tokens[current_index].clone(),
                        position: current_index,
                    });
                }
            };

            return Ok(self.push_expression(Expr::Assignment {
                identifier: identifier.clone(),
                value,
                op: token,
            }));
        };

        Ok(expr)
    }

    // logic_or     -> logic_and ( "or" logic_and )* ;
    fn logic_or(&mut self) -> Result<usize, ParseError> {
        let mut expr = self.logic_and()?;

        while self.matches(Token::Or) {
            let op = self.previous_token().clone();
            let rhs = self.logic_and()?;
            let lhs = expr;

            expr = self.push_expression(Expr::Binary { lhs, rhs, op });
        }

        Ok(expr)
    }

    // logic_and    -> equality ( "and" equality )* ;
    fn logic_and(&mut self) -> Result<usize, ParseError> {
        let mut expr = self.equality()?;

        while self.matches(Token::And) {
            let op = self.previous_token().clone();
            let rhs = self.logic_and()?;
            let lhs = expr;

            expr = self.push_expression(Expr::Binary { lhs, rhs, op });
        }

        Ok(expr)
    }

    // equality     -> comparison ( ( "!=" | "==" ) comparison )* ;
    fn equality(&mut self) -> Result<usize, ParseError> {
        let mut expr = self.comparison()?;

        while self.matches_any(&[Token::BangEqual, Token::EqualEqual]) {
            let op = self.previous_token().clone();
            let rhs = self.factor()?;
            let lhs = expr;

            expr = self.push_expression(Expr::Binary { lhs, rhs, op });
        }

        Ok(expr)
    }

    // comparison   -> term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
    fn comparison(&mut self) -> Result<usize, ParseError> {
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

            expr = self.push_expression(Expr::Binary { lhs, rhs, op });
        }

        Ok(expr)
    }

    // term         -> factor ( ( "-" | "+" ) factor )* ;
    fn term(&mut self) -> Result<usize, ParseError> {
        let mut expr = self.factor()?;

        while self.matches_any(&[Token::Minus, Token::Plus]) {
            let op = self.previous_token().clone();
            let rhs = self.factor()?;
            let lhs = expr;

            expr = self.push_expression(Expr::Binary { lhs, rhs, op });
        }

        Ok(expr)
    }

    // factor       -> unary ( ( "/" | "*" ) unary )* ;
    fn factor(&mut self) -> Result<usize, ParseError> {
        let mut expr = self.unary()?;

        while self.matches_any(&[Token::Asterisk, Token::Slash]) {
            let op = self.previous_token().clone();
            let rhs = self.unary()?;
            let lhs = expr;

            expr = self.push_expression(Expr::Binary { lhs, rhs, op });
        }

        Ok(expr)
    }

    // unary        -> ("not" | "-") unary
    //               | call ;
    fn unary(&mut self) -> Result<usize, ParseError> {
        if self.matches_any(&[Token::Minus, Token::Not]) {
            let op = self.previous_token().clone();
            let rhs = self.unary()?;

            return Ok(self.push_expression(Expr::Unary { rhs, op }));
        }

        self.call()
    }

    // call         -> primary ( "(" arguments? ")" "* ;
    // arguments    -> expression ( "," expression )* ","? ;
    fn call(&mut self) -> Result<usize, ParseError> {
        let start_index = self.current_index;
        let expr = self.primary()?;

        if self.matches(Token::OpenParen) {
            let callee = expr;

            match &self.expressions[callee] {
                Expr::Identifier(identifier) => {
                    return self.finish_call(identifier.clone());
                }
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: Token::Identifier("".to_string()),
                        found: self.tokens[start_index].clone(),
                        position: start_index,
                    });
                }
            };
        }

        Ok(expr)
    }

    fn finish_call(&mut self, callee: String) -> Result<usize, ParseError> {
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

        Ok(self.push_expression(Expr::Call { callee, arguments }))
    }

    // primary      -> NUMBER | STRING | "true" | "false" | "nil"
    //               | "(" expression ")" | IDENTIFIER;
    fn primary(&mut self) -> Result<usize, ParseError> {
        let expr = match self.current_token() {
            Token::Number(n) => self.push_expression(Expr::Number(*n)),
            Token::True => self.push_expression(Expr::Boolean(true)),
            Token::False => self.push_expression(Expr::Boolean(false)),
            Token::String(s) => self.push_expression(Expr::String(s.clone())),
            Token::Identifier(s) => self.push_expression(Expr::Identifier(s.clone())),
            Token::OpenParen => self.grouping()?,
            _ => {
                return Err(ParseError::InvalidTokenType {
                    found: self.current_token().clone(),
                    position: self.current_index,
                });
            }
        };

        self.advance();

        Ok(expr)
    }

    fn grouping(&mut self) -> Result<usize, ParseError> {
        self.consume(Token::OpenParen)?;
        let expr = self.expression()?;
        self.expect(Token::CloseParen)?;
        Ok(expr)
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

        false
    }

    fn matches_any(&mut self, tokens: &[Token]) -> bool {
        for token in tokens {
            if self.matches(token.clone()) {
                return true;
            }
        }

        false
    }

    fn check(&self, token: Token) -> bool {
        variant_eq(self.current_token(), &token)
    }

    fn consume(&mut self, expected: Token) -> Result<&Token, ParseError> {
        if variant_eq(self.current_token(), &expected) {
            return Ok(self.advance());
        }

        Err(ParseError::UnexpectedToken {
            expected: expected.clone(),
            found: self.current_token().clone(),
            position: self.current_index,
        })
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if !self.check(expected.clone()) {
            return Err(ParseError::UnexpectedToken {
                expected: expected.clone(),
                found: self.current_token().clone(),
                position: self.current_index,
            });
        }

        Ok(())
    }

    fn current_token(&self) -> &Token {
        &self.tokens[self.current_index]
    }

    fn previous_token(&self) -> &Token {
        &self.tokens[self.current_index - 1]
    }

    fn advance(&mut self) -> &Token {
        if self.current_index + 1 < self.tokens.len() {
            self.current_index += 1;
            return self.previous_token();
        }

        self.current_token()
    }

    fn push_statement(&mut self, stmt: Stmt) -> usize {
        let index = self.statements.len();
        self.statements.push(stmt);
        index
    }

    fn push_expression(&mut self, expr: Expr) -> usize {
        let index = self.expressions.len();
        self.expressions.push(expr);
        index
    }
}

#[cfg(test)]
mod stmt_tests {
    use crate::lexer::tokenize;

    use super::*;

    fn test_statement(
        input: &str,
        expected_stmt: Result<Option<usize>, ParseError>,
        expected_stmts: Vec<Stmt>,
        expected_exprs: Vec<Expr>,
    ) {
        let (tokens, errors) = tokenize(input);
        assert_eq!(errors, vec![]);

        let mut parser = Parser::new(tokens);
        let stmt = parser.declaration();

        assert_eq!(stmt, expected_stmt);
        assert_eq!(parser.statements, expected_stmts);
        assert_eq!(parser.expressions, expected_exprs);
    }

    #[test]
    fn empty_declaration() {
        test_statement("", Ok(None), vec![], vec![]);
    }

    #[test]
    fn expr_simple() {
        test_statement(
            "42;",
            Ok(Some(0)),
            vec![Stmt::Expr(0)],
            vec![Expr::Number(42.0)],
        );
    }

    #[test]
    fn expr_complex() {
        test_statement(
            "true or false and 42 < 43;",
            Ok(Some(0)),
            vec![Stmt::Expr(6)],
            vec![
                Expr::Boolean(true),
                Expr::Boolean(false),
                Expr::Number(42.0),
                Expr::Number(43.0),
                Expr::Binary {
                    lhs: 2,
                    rhs: 3,
                    op: Token::Less,
                },
                Expr::Binary {
                    lhs: 1,
                    rhs: 4,
                    op: Token::And,
                },
                Expr::Binary {
                    lhs: 0,
                    rhs: 5,
                    op: Token::Or,
                },
            ],
        );
    }

    #[test]
    fn var_decl_no_value() {
        test_statement(
            "var foo;",
            Ok(Some(0)),
            vec![Stmt::VarDecl {
                identifier: "foo".to_string(),
                expr: None,
            }],
            vec![],
        );
    }

    #[test]
    fn var_decl_string_value() {
        test_statement(
            "var toto = \"tata\";",
            Ok(Some(0)),
            vec![Stmt::VarDecl {
                identifier: "toto".to_string(),
                expr: Some(0),
            }],
            vec![Expr::String("tata".to_string())],
        );
    }

    #[test]
    fn var_decl_missing_semicolon() {
        test_statement(
            "var bar",
            Err(ParseError::UnexpectedToken {
                expected: Token::Semicolon,
                found: Token::EOF,
                position: 2,
            }),
            vec![],
            vec![],
        );
    }

    #[test]
    fn var_decl_with_value_missing_semicolon() {
        test_statement(
            "var baz = 42",
            Err(ParseError::UnexpectedToken {
                expected: Token::Semicolon,
                found: Token::EOF,
                position: 4,
            }),
            vec![],
            vec![Expr::Number(42.0)],
        );
    }

    #[test]
    fn empty_block() {
        test_statement("{}", Ok(Some(0)), vec![Stmt::Block(vec![])], vec![]);
    }

    #[test]
    fn block_with_one_statement() {
        test_statement(
            "{42;}",
            Ok(Some(1)),
            vec![Stmt::Expr(0), Stmt::Block(vec![0])],
            vec![Expr::Number(42.0)],
        );
    }

    #[test]
    fn block_with_multiple_statements() {
        test_statement(
            r#"{
                    var foo = 42;
                    var bar = 1337;
                    var baz = foo + bar;
                }"#,
            Ok(Some(3)),
            vec![
                Stmt::VarDecl {
                    identifier: "foo".to_string(),
                    expr: Some(0),
                },
                Stmt::VarDecl {
                    identifier: "bar".to_string(),
                    expr: Some(1),
                },
                Stmt::VarDecl {
                    identifier: "baz".to_string(),
                    expr: Some(4),
                },
                Stmt::Block(vec![0, 1, 2]),
            ],
            vec![
                Expr::Number(42.0),
                Expr::Number(1337.0),
                Expr::Identifier("foo".to_string()),
                Expr::Identifier("bar".to_string()),
                Expr::Binary {
                    lhs: 2,
                    rhs: 3,
                    op: Token::Plus,
                },
            ],
        );
    }

    #[test]
    fn block_error_missing_brace() {
        test_statement(
            "{ 42; ",
            Err(ParseError::UnexpectedEndOfFile {
                expected: Token::CloseBrace,
                position: 0,
            }),
            vec![Stmt::Expr(0)],
            vec![Expr::Number(42.0)],
        );
    }

    #[test]
    fn function_decl_1() {
        test_statement(
            "fn foo() {}",
            Ok(Some(1)),
            vec![
                Stmt::Block(vec![]),
                Stmt::FunctionDecl {
                    identifier: "foo".to_string(),
                    args: vec![],
                    body: 0,
                },
            ],
            vec![],
        );
    }

    #[test]
    fn function_decl_2() {
        test_statement(
            "fn foo(a) {}",
            Ok(Some(1)),
            vec![
                Stmt::Block(vec![]),
                Stmt::FunctionDecl {
                    identifier: "foo".to_string(),
                    args: vec!["a".to_string()],
                    body: 0,
                },
            ],
            vec![],
        );
    }

    #[test]
    fn function_decl_3() {
        test_statement(
            "fn foo(bar, baz) {}",
            Ok(Some(1)),
            vec![
                Stmt::Block(vec![]),
                Stmt::FunctionDecl {
                    identifier: "foo".to_string(),
                    args: vec!["bar".to_string(), "baz".to_string()],
                    body: 0,
                },
            ],
            vec![],
        );
    }

    #[test]
    fn function_decl_4() {
        test_statement(
            "fn foo(bar, baz {}",
            Err(ParseError::UnexpectedToken {
                expected: Token::CloseParen,
                found: Token::OpenBrace,
                position: 6,
            }),
            vec![],
            vec![],
        );
    }

    #[test]
    fn if_stmt_1() {
        test_statement(
            "if true {}",
            Ok(Some(1)),
            vec![
                Stmt::Block(vec![]),
                Stmt::If {
                    cond: 0,
                    if_block: 0,
                    else_block: None,
                },
            ],
            vec![Expr::Boolean(true)],
        );
    }

    #[test]
    fn if_stmt_2() {
        test_statement(
            "if false {} else {}",
            Ok(Some(2)),
            vec![
                Stmt::Block(vec![]),
                Stmt::Block(vec![]),
                Stmt::If {
                    cond: 0,
                    if_block: 0,
                    else_block: Some(1),
                },
            ],
            vec![Expr::Boolean(false)],
        );
    }

    #[test]
    fn if_stmt_3() {
        test_statement(
            "if false {} else if true {}",
            Ok(Some(3)),
            vec![
                Stmt::Block(vec![]),
                Stmt::Block(vec![]),
                Stmt::If {
                    cond: 1,
                    if_block: 1,
                    else_block: None,
                },
                Stmt::If {
                    cond: 0,
                    if_block: 0,
                    else_block: Some(2),
                },
            ],
            vec![Expr::Boolean(false), Expr::Boolean(true)],
        );
    }

    #[test]
    fn if_stmt_4() {
        test_statement(
            "if false {} else if true {} else if false {} else {}",
            Ok(Some(6)),
            vec![
                Stmt::Block(vec![]),
                Stmt::Block(vec![]),
                Stmt::Block(vec![]),
                Stmt::Block(vec![]),
                Stmt::If {
                    cond: 2,
                    if_block: 2,
                    else_block: Some(3),
                },
                Stmt::If {
                    cond: 1,
                    if_block: 1,
                    else_block: Some(4),
                },
                Stmt::If {
                    cond: 0,
                    if_block: 0,
                    else_block: Some(5),
                },
            ],
            vec![
                Expr::Boolean(false),
                Expr::Boolean(true),
                Expr::Boolean(false),
            ],
        );
    }

    #[test]
    fn if_stmt_5() {
        test_statement(
            "if false else {}",
            Err(ParseError::UnexpectedToken {
                expected: Token::OpenBrace,
                found: Token::Else,
                position: 2,
            }),
            vec![],
            vec![Expr::Boolean(false)],
        );
    }

    #[test]
    fn if_stmt_6() {
        test_statement(
            "if false {} else foo;",
            Err(ParseError::Todo),
            vec![Stmt::Block(vec![])],
            vec![Expr::Boolean(false)],
        );
    }

    #[test]
    fn while_stmt_1() {
        test_statement(
            "while true {}",
            Ok(Some(1)),
            vec![Stmt::Block(vec![]), Stmt::While { cond: 0, block: 0 }],
            vec![Expr::Boolean(true)],
        );
    }

    #[test]
    fn while_stmt_2() {
        test_statement(
            "while false;",
            Err(ParseError::UnexpectedToken {
                expected: Token::OpenBrace,
                found: Token::Semicolon,
                position: 2,
            }),
            vec![],
            vec![Expr::Boolean(false)],
        );
    }

    #[test]
    fn for_empty() {
        test_statement(
            "for ;; {}",
            Ok(Some(2)),
            vec![
                Stmt::Block(vec![]),
                Stmt::While { cond: 0, block: 0 },
                Stmt::Block(vec![1]),
            ],
            vec![Expr::Boolean(true)],
        );
    }

    #[test]
    fn for_assign_init() {
        test_statement(
            "for i = 0;; {}",
            Ok(Some(3)),
            vec![
                Stmt::Expr(2),
                Stmt::Block(vec![]),
                Stmt::While { cond: 3, block: 1 },
                Stmt::Block(vec![0, 2]),
            ],
            vec![
                Expr::Identifier("i".to_string()),
                Expr::Number(0.0),
                Expr::Assignment {
                    identifier: "i".to_string(),
                    value: 1,
                    op: Token::Equal,
                },
                Expr::Boolean(true),
            ],
        );
    }

    #[test]
    fn for_var_decl_init() {
        test_statement(
            "for var i = 0;; {}",
            Ok(Some(3)),
            vec![
                Stmt::VarDecl {
                    identifier: "i".to_string(),
                    expr: Some(0),
                },
                Stmt::Block(vec![]),
                Stmt::While { cond: 1, block: 1 },
                Stmt::Block(vec![0, 2]),
            ],
            vec![Expr::Number(0.0), Expr::Boolean(true)],
        );
    }

    #[test]
    fn for_complete() {
        test_statement(
            "for var i = 0; i < 10; i = i + 1 { print(i); }",
            Ok(Some(5)),
            vec![
                Stmt::VarDecl {
                    identifier: "i".to_string(),
                    expr: Some(0),
                },
                Stmt::Expr(11),
                Stmt::Block(vec![1, 3]),
                Stmt::Expr(8),
                Stmt::While { cond: 3, block: 2 },
                Stmt::Block(vec![0, 4]),
            ],
            vec![
                Expr::Number(0.0),
                Expr::Identifier("i".to_string()),
                Expr::Number(10.0),
                Expr::Binary {
                    lhs: 1,
                    rhs: 2,
                    op: Token::Less,
                },
                Expr::Identifier("i".to_string()),
                Expr::Identifier("i".to_string()),
                Expr::Number(1.0),
                Expr::Binary {
                    lhs: 5,
                    rhs: 6,
                    op: Token::Plus,
                },
                Expr::Assignment {
                    identifier: "i".to_string(),
                    value: 7,
                    op: Token::Equal,
                },
                Expr::Identifier("print".to_string()),
                Expr::Identifier("i".to_string()),
                Expr::Call {
                    callee: "print".to_string(),
                    arguments: vec![10],
                },
            ],
        );
    }

    #[test]
    fn return_empty() {
        test_statement("return;", Ok(Some(0)), vec![Stmt::Return(None)], vec![]);
    }

    #[test]
    fn return_expr() {
        test_statement(
            "return 42;",
            Ok(Some(0)),
            vec![Stmt::Return(Some(0))],
            vec![Expr::Number(42.0)],
        );
    }

    // TODO: Test program
}

#[cfg(test)]
mod expr_tests {
    use crate::lexer::tokenize;

    use super::*;

    fn test_expression(input: &str, expected: Result<Vec<Expr>, ParseError>) {
        let (tokens, errors) = tokenize(input);
        assert_eq!(errors, vec![]);

        let mut parser = Parser::new(tokens);
        let expr = parser.expression();

        match expr {
            Ok(_) => assert_eq!(parser.expressions, expected.unwrap()),
            Err(err) => assert_eq!(err, expected.unwrap_err()),
        }
    }

    #[test]
    fn primary_number_1() {
        test_expression("42", Ok(vec![Expr::Number(42.0)]));
    }

    #[test]
    fn primary_number_2() {
        test_expression("13.37", Ok(vec![Expr::Number(13.37)]));
    }

    #[test]
    fn primary_bool_true() {
        test_expression("true", Ok(vec![Expr::Boolean(true)]));
    }

    #[test]
    fn primary_bool_false() {
        test_expression("false", Ok(vec![Expr::Boolean(false)]));
    }

    #[test]
    fn primary_string() {
        test_expression(
            "\"hello, world\"",
            Ok(vec![Expr::String("hello, world".into())]),
        );
    }

    #[test]
    fn primary_invalid() {
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
        test_expression("(42)", Ok(vec![Expr::Number(42.0)]));
        test_expression(
            "1 * (2 + 3)",
            Ok(vec![
                Expr::Number(1.0),
                Expr::Number(2.0),
                Expr::Number(3.0),
                Expr::Binary {
                    lhs: 1,
                    rhs: 2,
                    op: Token::Plus,
                },
                Expr::Binary {
                    lhs: 0,
                    rhs: 3,
                    op: Token::Asterisk,
                },
            ]),
        );
    }

    #[test]
    fn unary_minus() {
        test_expression(
            "-42",
            Ok(vec![
                Expr::Number(42.0),
                Expr::Unary {
                    rhs: 0,
                    op: Token::Minus,
                },
            ]),
        );
    }

    #[test]
    fn unary_not() {
        test_expression(
            "not false",
            Ok(vec![
                Expr::Boolean(false),
                Expr::Unary {
                    rhs: 0,
                    op: Token::Not,
                },
            ]),
        );
    }

    #[test]
    fn term_add() {
        test_expression(
            "1 + 2",
            Ok(vec![
                Expr::Number(1.0),
                Expr::Number(2.0),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::Plus,
                },
            ]),
        );
    }

    #[test]
    fn term_substract() {
        test_expression(
            "4.2 - 13.37",
            Ok(vec![
                Expr::Number(4.2),
                Expr::Number(13.37),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::Minus,
                },
            ]),
        );
    }

    #[test]
    fn factor_multiply() {
        test_expression(
            "2 * 3",
            Ok(vec![
                Expr::Number(2.0),
                Expr::Number(3.0),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::Asterisk,
                },
            ]),
        );
    }

    #[test]
    fn factor_divide() {
        test_expression(
            "7 / 4",
            Ok(vec![
                Expr::Number(7.0),
                Expr::Number(4.0),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::Slash,
                },
            ]),
        );
    }

    #[test]
    fn logic_or() {
        test_expression(
            "true or false",
            Ok(vec![
                Expr::Boolean(true),
                Expr::Boolean(false),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::Or,
                },
            ]),
        );
    }

    #[test]
    fn logic_and() {
        test_expression(
            "false and true",
            Ok(vec![
                Expr::Boolean(false),
                Expr::Boolean(true),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::And,
                },
            ]),
        );
    }

    #[test]
    fn equality_equal() {
        test_expression(
            "33.0 == false",
            Ok(vec![
                Expr::Number(33.0),
                Expr::Boolean(false),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::EqualEqual,
                },
            ]),
        );
    }

    #[test]
    fn equality_not_equal() {
        test_expression(
            "\"hello\" != \"test\"",
            Ok(vec![
                Expr::String("hello".into()),
                Expr::String("test".into()),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::BangEqual,
                },
            ]),
        );
    }

    #[test]
    fn comparison_gt() {
        test_expression(
            "42 > 33",
            Ok(vec![
                Expr::Number(42.0),
                Expr::Number(33.0),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::Greater,
                },
            ]),
        );
    }

    #[test]
    fn comparison_ge() {
        test_expression(
            "42 >= 33",
            Ok(vec![
                Expr::Number(42.0),
                Expr::Number(33.0),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::GreaterEqual,
                },
            ]),
        );
    }

    #[test]
    fn comparison_lt() {
        test_expression(
            "42 < 33",
            Ok(vec![
                Expr::Number(42.0),
                Expr::Number(33.0),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::Less,
                },
            ]),
        );
    }

    #[test]
    fn comparison_le() {
        test_expression(
            "42 <= 33",
            Ok(vec![
                Expr::Number(42.0),
                Expr::Number(33.0),
                Expr::Binary {
                    lhs: 0,
                    rhs: 1,
                    op: Token::LessEqual,
                },
            ]),
        );
    }

    #[test]
    fn call_no_args() {
        test_expression(
            "foo()",
            Ok(vec![
                Expr::Identifier("foo".to_string()),
                Expr::Call {
                    callee: "foo".to_string(),
                    arguments: vec![],
                },
            ]),
        );
    }

    #[test]
    fn call_multiple_args() {
        test_expression(
            "bar(\"test\", 39 + 3)",
            Ok(vec![
                Expr::Identifier("bar".to_string()),
                Expr::String("test".to_string()),
                Expr::Number(39.0),
                Expr::Number(3.0),
                Expr::Binary {
                    lhs: 2,
                    rhs: 3,
                    op: Token::Plus,
                },
                Expr::Call {
                    callee: "bar".to_string(),
                    arguments: vec![1, 4],
                },
            ]),
        );
    }

    #[test]
    fn call_invalid_name() {
        test_expression(
            "4()",
            Err(ParseError::UnexpectedToken {
                expected: Token::Identifier("".to_string()),
                found: Token::Number(4.0),
                position: 0,
            }),
        );
    }

    #[test]
    fn call_missing_close_paren() {
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
    fn assign_number() {
        test_expression(
            "a = 42",
            Ok(vec![
                Expr::Identifier("a".to_string()),
                Expr::Number(42.0),
                Expr::Assignment {
                    identifier: "a".to_string(),
                    value: 1,
                    op: Token::Equal,
                },
            ]),
        );
    }

    #[test]
    fn assign_binary() {
        test_expression(
            "foo=15+27",
            Ok(vec![
                Expr::Identifier("foo".to_string()),
                Expr::Number(15.0),
                Expr::Number(27.0),
                Expr::Binary {
                    lhs: 1,
                    rhs: 2,
                    op: Token::Plus,
                },
                Expr::Assignment {
                    identifier: "foo".to_string(),
                    value: 3,
                    op: Token::Equal,
                },
            ]),
        );
    }

    #[test]
    fn assign_not_an_identifier() {
        test_expression(
            "27 = 42",
            Err(ParseError::UnexpectedToken {
                expected: Token::Identifier("".to_string()),
                found: Token::Number(27.0),
                position: 0,
            }),
        );
    }

    #[test]
    fn assign_plus_equal() {
        test_expression(
            "a += 2",
            Ok(vec![
                Expr::Identifier("a".to_string()),
                Expr::Number(2.0),
                Expr::Assignment {
                    identifier: "a".to_string(),
                    value: 1,
                    op: Token::PlusEqual,
                },
            ]),
        );
    }

    #[test]
    fn assign_minus_equal() {
        test_expression(
            "a -= 2",
            Ok(vec![
                Expr::Identifier("a".to_string()),
                Expr::Number(2.0),
                Expr::Assignment {
                    identifier: "a".to_string(),
                    value: 1,
                    op: Token::MinusEqual,
                },
            ]),
        );
    }

    #[test]
    fn assign_times_equal() {
        test_expression(
            "a *= 2",
            Ok(vec![
                Expr::Identifier("a".to_string()),
                Expr::Number(2.0),
                Expr::Assignment {
                    identifier: "a".to_string(),
                    value: 1,
                    op: Token::AsteriskEqual,
                },
            ]),
        );
    }

    #[test]
    fn assign_slash_equal() {
        test_expression(
            "a /= 2",
            Ok(vec![
                Expr::Identifier("a".to_string()),
                Expr::Number(2.0),
                Expr::Assignment {
                    identifier: "a".to_string(),
                    value: 1,
                    op: Token::SlashEqual,
                },
            ]),
        );
    }
}
