//! `rej-1b`: train a Rej model and generate with concept controls.

mod cli;

fn main() -> std::process::ExitCode {
    match cli::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rej-1b: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
