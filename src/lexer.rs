use phf::phf_map;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Token<'a> {
    Invalid(&'a str),

    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,

    Semicolon,
    Comma,
    Dot,

    Plus,
    Minus,
    Asterisk,
    Slash,
    PlusEqual,
    MinusEqual,
    AsteriskEqual,
    SlashEqual,
    Equal,
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    Number(&'a str),
    Identifier(&'a str),
    String(&'a str),

    Var,
    Nil,
    And,
    Or,
    Not,
    If,
    Else,
    For,
    While,
    True,
    False,
    Struct,
    Super,
    This,
    Fun,
    Return,
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
    "fun" => Token::Fun,
    "return" => Token::Return,
};

pub struct Lexer<'a> {
    src: &'a str,
    start_index: usize,
    current_index: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src,
            start_index: 0,
            current_index: 0,
        }
    }

    fn invalid(&self) -> Token<'a> {
        Token::Invalid(&self.src[self.start_index..self.current_index])
    }

    fn char_equal_token(&mut self, one_char: Token<'a>, two_char: Token<'a>) -> Option<Token<'a>> {
        if self.matches('=') {
            Some(two_char)
        } else {
            Some(one_char)
        }
    }

    fn identifier(&mut self, curr: char) -> Option<Token<'a>> {
        if !curr.is_alphanumeric() && curr != '_' {
            return Some(self.invalid());
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

        if let Some(&keyword) = KEYWORDS.get(&self.src[self.start_index..self.current_index]) {
            Some(keyword)
        } else {
            Some(Token::Identifier(
                &self.src[self.start_index..self.current_index],
            ))
        }
    }

    fn number(&mut self) -> Option<Token<'a>> {
        while self.peek().unwrap_or('\0').is_digit(10) {
            self.advance();
        }

        if self.peek().unwrap_or('\0') == '.' && self.peek_next().unwrap_or('\0').is_digit(10) {
            self.advance();

            while self.peek().unwrap_or('\0').is_digit(10) {
                self.advance();
            }
        }

        Some(Token::Number(
            &self.src[self.start_index..self.current_index],
        ))
    }

    fn string(&mut self) -> Option<Token<'a>> {
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
                return Some(self.invalid());
            }
        }

        Some(Token::String(
            &self.src[self.start_index + 1..self.current_index - 1],
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
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_whitespace();

        self.start_index = self.current_index;

        let char = self.advance();

        if let Some(char) = char {
            match char {
                '(' => Some(Token::OpenParen),
                ')' => Some(Token::CloseParen),
                '{' => Some(Token::OpenBrace),
                '}' => Some(Token::CloseBrace),
                '[' => Some(Token::OpenBracket),
                ']' => Some(Token::CloseBracket),
                ';' => Some(Token::Semicolon),
                ',' => Some(Token::Comma),
                '.' => Some(Token::Dot),
                '+' => self.char_equal_token(Token::Plus, Token::PlusEqual),
                '-' => self.char_equal_token(Token::Minus, Token::MinusEqual),
                '*' => self.char_equal_token(Token::Asterisk, Token::AsteriskEqual),
                '/' => self.char_equal_token(Token::Slash, Token::SlashEqual),
                '=' => self.char_equal_token(Token::Equal, Token::EqualEqual),
                '!' => self.char_equal_token(self.invalid(), Token::BangEqual),
                '<' => self.char_equal_token(Token::Less, Token::LessEqual),
                '>' => self.char_equal_token(Token::Greater, Token::GreaterEqual),
                '"' => self.string(),
                '0'..'9' => self.number(),
                _ => self.identifier(char),
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn punctuators() {
        let lexer = Lexer::new("(){}[];,+-*!!====<=>=<>/.+=-=*=/=");
        let found = lexer.collect_vec();

        let expected = vec![
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
            Token::Invalid("!"),
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
        ];

        assert_eq!(found, expected);
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

        let lexer = Lexer::new(str);
        let found = lexer.collect_vec();

        let expected = vec![
            Token::Identifier("space"),
            Token::Identifier("tabs"),
            Token::Identifier("newlines"),
            Token::Identifier("comments"),
            Token::Identifier("end"),
        ];

        assert_eq!(found, expected);
    }

    #[test]
    fn numbers() {
        let lexer = Lexer::new("123 123.456 .456 123.");
        let found = lexer.collect_vec();

        let expected = vec![
            Token::Number("123"),
            Token::Number("123.456"),
            Token::Dot,
            Token::Number("456"),
            Token::Number("123"),
            Token::Dot,
        ];

        assert_eq!(found, expected);
    }

    #[test]
    fn identifiers() {
        let str = r#"
andy formless fo _ _123 _abc ab123
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_
        "#;
        let lexer = Lexer::new(str);

        let found = lexer.collect_vec();

        let expected = vec![
            Token::Identifier("andy"),
            Token::Identifier("formless"),
            Token::Identifier("fo"),
            Token::Identifier("_"),
            Token::Identifier("_123"),
            Token::Identifier("_abc"),
            Token::Identifier("ab123"),
            Token::Identifier("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_"),
        ];

        assert_eq!(found, expected);
    }

    #[test]
    fn keywords() {
        let str = r#"
and struct else false for fun if nil not or return super this true var while 
        "#;

        let lexer = Lexer::new(str);
        let found = lexer.collect_vec();

        let expected = vec![
            Token::And,
            Token::Struct,
            Token::Else,
            Token::False,
            Token::For,
            Token::Fun,
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
        ];

        assert_eq!(found, expected);
    }

    #[test]
    fn strings() {
        let str = r#"
        "" 
        "string" 
        "string \"with escaped quotes\""
        "unfinished string"#;
        let lexer = Lexer::new(str);
        let found = lexer.collect_vec();

        let expected = vec![
            Token::String(""),
            Token::String("string"),
            Token::String("string \\\"with escaped quotes\\\""),
            Token::Invalid("\"unfinished string"),
        ];

        assert_eq!(found, expected);
    }

    #[test]
    fn misc() {
        let lexer = Lexer::new("π Hello, World if for While while true false; 1337 42 893.17");

        let found = lexer.collect_vec();

        let expected = vec![
            Token::Identifier("π"),
            Token::Identifier("Hello"),
            Token::Comma,
            Token::Identifier("World"),
            Token::If,
            Token::For,
            Token::Identifier("While"),
            Token::While,
            Token::True,
            Token::False,
            Token::Semicolon,
            Token::Number("1337"),
            Token::Number("42"),
            Token::Number("893.17"),
        ];

        assert_eq!(found, expected);
    }
}
