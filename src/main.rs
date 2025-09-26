use charm_rs::interpret_program;

fn main() {
    if std::env::args().len() <= 1 {
        println!("Missing file input");
        return;
    }

    let input = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();

    interpret_program(&input);
}
