use super::token::Token;
use itertools::{Either, Itertools};
use phf::phf_map;

#[derive(Debug, Clone, PartialEq)]
pub enum LexError {
    UnexpectedCharacter { char: char, position: usize },
    UnterminatedString { position: usize },
    InvalidNumber { text: String, position: usize },
}

static KEYWORDS: phf::Map<&'static str, Token> = phf_map! {
    "var" => Token::Var,
    "nil" => Token::Nil,
    "and" => Token::And,
    "or" => Token::Or,
    "not" => Token::Not,
    "if" => Token::If,
    "else" => Token::Else,
    "for" => Token::For,
    "while" => Token::While,
    "true" => Token::True,
    "false" => Token::False,
    "struct" => Token::Struct,
    "super" => Token::Super,
    "this" => Token::This,
    "fn" => Token::Function,
    "return" => Token::Return,
};

pub fn tokenize(src: &str) -> (Vec<Token>, Vec<LexError>) {
    let (mut tokens, errors): (Vec<Token>, Vec<LexError>) =
        Lexer::new(src).partition_map(|result| match result {
            Ok(token) => Either::Left(token),
            Err(e) => Either::Right(e),
        });

    tokens.push(Token::EOF);

    return (tokens, errors);
}

struct Lexer<'a> {
    src: &'a str,
    start_index: usize,
    current_index: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src,
            start_index: 0,
            current_index: 0,
        }
    }

    fn char_equal_token(&mut self, one_char: Token, two_char: Token) -> Result<Token, LexError> {
        if self.matches('=') {
            return Ok(two_char);
        }

        return Ok(one_char);
    }

    fn identifier(&mut self, curr: char) -> Result<Token, LexError> {
        if !curr.is_alphanumeric() && curr != '_' {
            return Err(LexError::UnexpectedCharacter {
                char: curr,
                position: self.start_index,
            });
        }

        loop {
            if let Some(c) = self.peek() {
                if c.is_alphanumeric() || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if let Some(keyword) = KEYWORDS.get(&self.src[self.start_index..self.current_index]) {
            return Ok(keyword.clone());
        }

        return Ok(Token::Identifier(
            self.src[self.start_index..self.current_index].to_string(),
        ));
    }

    fn number(&mut self) -> Result<Token, LexError> {
        while self.peek().unwrap_or('\0').is_digit(10) {
            self.advance();
        }

        if self.peek().unwrap_or('\0') == '.' && self.peek_next().unwrap_or('\0').is_digit(10) {
            self.advance();

            while self.peek().unwrap_or('\0').is_digit(10) {
                self.advance();
            }
        }

        let str = &self.src[self.start_index..self.current_index];

        return match str.parse::<f64>() {
            Ok(number) => Ok(Token::Number(number)),
            Err(_) => Err(LexError::InvalidNumber {
                text: str.to_string(),
                position: self.start_index,
            }),
        };
    }

    fn string(&mut self) -> Result<Token, LexError> {
        loop {
            if let Some(c) = self.peek() {
                match c {
                    '\\' => {
                        self.consume();
                        self.consume();
                    }
                    '"' => {
                        self.consume();
                        break;
                    }
                    _ => self.consume(),
                }
            } else {
                return Err(LexError::UnterminatedString {
                    position: self.start_index,
                });
            }
        }

        Ok(Token::String(
            self.src[self.start_index + 1..self.current_index - 1].to_string(),
        ))
    }

    fn skip_whitespace(&mut self) {
        loop {
            match self.peek().unwrap_or('\0') {
                ' ' | '\t' | '\n' | '\r' => {
                    self.consume();
                }
                '/' => {
                    if self.peek_next().unwrap_or('\0') == '/' {
                        self.consume();
                        self.consume();
                        loop {
                            match self.peek() {
                                Some(c) => match c {
                                    '\n' => break,
                                    _ => self.consume(),
                                },
                                None => break,
                            }
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
    }

    fn advance(&mut self) -> Option<char> {
        if self.current_index >= self.src.len() {
            return None;
        }

        let char = self.src[self.current_index..]
            .chars()
            .peekable()
            .peek()
            .copied();

        self.current_index += char?.len_utf8();

        return char;
    }

    fn consume(&mut self) {
        self.advance();
    }

    fn peek(&self) -> Option<char> {
        if self.current_index >= self.src.len() {
            return None;
        }

        self.src[self.current_index..]
            .chars()
            .peekable()
            .peek()
            .copied()
    }

    fn peek_next(&self) -> Option<char> {
        if (self.current_index + 1) >= self.src.len() {
            return None;
        }

        self.src[self.current_index + 1..]
            .chars()
            .peekable()
            .peek()
            .copied()
    }

    fn matches(&mut self, c: char) -> bool {
        if self.peek().unwrap_or('\0') == c {
            self.advance();
            true
        } else {
            false
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token, LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_whitespace();

        self.start_index = self.current_index;

        let char = self.advance();

        if let Some(char) = char {
            match char {
                '(' => Some(Ok(Token::OpenParen)),
                ')' => Some(Ok(Token::CloseParen)),
                '{' => Some(Ok(Token::OpenBrace)),
                '}' => Some(Ok(Token::CloseBrace)),
                '[' => Some(Ok(Token::OpenBracket)),
                ']' => Some(Ok(Token::CloseBracket)),
                ';' => Some(Ok(Token::Semicolon)),
                ',' => Some(Ok(Token::Comma)),
                '.' => Some(Ok(Token::Dot)),
                '+' => Some(self.char_equal_token(Token::Plus, Token::PlusEqual)),
                '-' => Some(self.char_equal_token(Token::Minus, Token::MinusEqual)),
                '*' => Some(self.char_equal_token(Token::Asterisk, Token::AsteriskEqual)),
                '/' => Some(self.char_equal_token(Token::Slash, Token::SlashEqual)),
                '=' => Some(self.char_equal_token(Token::Equal, Token::EqualEqual)),
                '<' => Some(self.char_equal_token(Token::Less, Token::LessEqual)),
                '>' => Some(self.char_equal_token(Token::Greater, Token::GreaterEqual)),
                '!' => {
                    if self.matches('=') {
                        Some(Ok(Token::BangEqual))
                    } else {
                        Some(Err(LexError::UnexpectedCharacter {
                            char,
                            position: self.start_index,
                        }))
                    }
                }
                '"' => Some(self.string()),
                '0'..'9' => Some(self.number()),
                _ => Some(self.identifier(char)),
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuators() {
        let (tokens, errors) = tokenize("(){}[];,+-*!!====<=>=<>/.+=-=*=/=");

        let expected_tokens = vec![
            Token::OpenParen,
            Token::CloseParen,
            Token::OpenBrace,
            Token::CloseBrace,
            Token::OpenBracket,
            Token::CloseBracket,
            Token::Semicolon,
            Token::Comma,
            Token::Plus,
            Token::Minus,
            Token::Asterisk,
            Token::BangEqual,
            Token::EqualEqual,
            Token::Equal,
            Token::LessEqual,
            Token::GreaterEqual,
            Token::Less,
            Token::Greater,
            Token::Slash,
            Token::Dot,
            Token::PlusEqual,
            Token::MinusEqual,
            Token::AsteriskEqual,
            Token::SlashEqual,
            Token::EOF,
        ];

        let expected_errors = vec![LexError::UnexpectedCharacter {
            char: '!',
            position: 11,
        }];

        assert_eq!(tokens, expected_tokens);
        assert_eq!(errors, expected_errors);
    }

    #[test]
    fn whitespace() {
        let str = r#"
space    tabs				newlines



comments // comments !
// On 
// multiple lines

end
        "#;

        let (tokens, errors) = tokenize(str);

        let expected = vec![
            Token::Identifier("space".to_string()),
            Token::Identifier("tabs".to_string()),
            Token::Identifier("newlines".to_string()),
            Token::Identifier("comments".to_string()),
            Token::Identifier("end".to_string()),
            Token::EOF,
        ];

        assert_eq!(tokens, expected);
        assert_eq!(errors, vec![]);
    }

    #[test]
    fn numbers() {
        let (tokens, errors) = tokenize("123 123.456 .456 123.");

        let expected = vec![
            Token::Number(123.0),
            Token::Number(123.456),
            Token::Dot,
            Token::Number(456.0),
            Token::Number(123.0),
            Token::Dot,
            Token::EOF,
        ];

        assert_eq!(tokens, expected);
        assert_eq!(errors, vec![]);
    }

    #[test]
    fn identifiers() {
        let str = r#"
andy formless fo _ _123 _abc ab123
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_
        "#;
        let (tokens, errors) = tokenize(str);

        let expected = vec![
            Token::Identifier("andy".to_string()),
            Token::Identifier("formless".to_string()),
            Token::Identifier("fo".to_string()),
            Token::Identifier("_".to_string()),
            Token::Identifier("_123".to_string()),
            Token::Identifier("_abc".to_string()),
            Token::Identifier("ab123".to_string()),
            Token::Identifier(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_".to_string(),
            ),
            Token::EOF,
        ];

        assert_eq!(tokens, expected);
        assert_eq!(errors, vec![]);
    }

    #[test]
    fn keywords() {
        let str = r#"
and struct else false for fn if nil not or return super this true var while 
        "#;

        let (tokens, errors) = tokenize(str);

        let expected = vec![
            Token::And,
            Token::Struct,
            Token::Else,
            Token::False,
            Token::For,
            Token::Function,
            Token::If,
            Token::Nil,
            Token::Not,
            Token::Or,
            Token::Return,
            Token::Super,
            Token::This,
            Token::True,
            Token::Var,
            Token::While,
            Token::EOF,
        ];

        assert_eq!(tokens, expected);
        assert_eq!(errors, vec![]);
    }

    #[test]
    fn strings() {
        let str = r#"
        "" 
        "string" 
        "string \"with escaped quotes\""
        "unfinished string"#;

        let (tokens, errors) = tokenize(str);

        let expected_tokens = vec![
            Token::String("".to_string()),
            Token::String("string".to_string()),
            Token::String("string \\\"with escaped quotes\\\"".to_string()),
            Token::EOF,
        ];

        let expected_errors = vec![LexError::UnterminatedString { position: 80 }];

        assert_eq!(tokens, expected_tokens);
        assert_eq!(errors, expected_errors);
    }

    #[test]
    fn misc() {
        let (tokens, errors) =
            tokenize("π Hello, World if for While while true false; 1337 42 893.17");

        let expected = vec![
            Token::Identifier("π".to_string()),
            Token::Identifier("Hello".to_string()),
            Token::Comma,
            Token::Identifier("World".to_string()),
            Token::If,
            Token::For,
            Token::Identifier("While".to_string()),
            Token::While,
            Token::True,
            Token::False,
            Token::Semicolon,
            Token::Number(1337.0),
            Token::Number(42.0),
            Token::Number(893.17),
            Token::EOF,
        ];

        assert_eq!(tokens, expected);
        assert_eq!(errors, vec![]);
    }
}
