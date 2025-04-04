use nix::sys::signal::{self, SaFlags, SigAction, SigHandler, SigSet, Signal};
use std::env;
use std::fs::OpenOptions;
use std::io::Result;
use std::io::Write;
use std::process::Command;

extern "C" fn handle_sigchld(_: i32) {}

fn log_to_file(message: String) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true) // Create the file if it doesn't exist
        .append(true) // Append to the file if it exists
        .open("/tmp/trace.log")?; // Open or create the log file

    writeln!(file, "{}", message)?; // Write the message to the file with a newline
    Ok(())
}

fn unescape_string(input: &str) -> String {
    input.replace("\\=", "=").replace("\\;", ";")
}

fn deserialize_env_vars(serialized: &str) -> Vec<(String, String)> {
    serialized
        .split(';')
        .filter_map(|pair| {
            let mut iter = pair.splitn(2, '=');
            let key = iter.next()?;
            let value = iter.next()?;
            Some((unescape_string(key), unescape_string(value)))
        })
        .collect()
}

fn main() {
    let sig_action = SigAction::new(
        SigHandler::Handler(handle_sigchld),
        SaFlags::empty(),
        SigSet::empty(),
    );
    unsafe {
        signal::sigaction(Signal::SIGCHLD, &sig_action).expect("Failed to set SIGCHLD handler");
    }

    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        panic!("Error: Not enough arguments provided.");
    }

    let serialized_env_vars = &args[0];
    let command = &args[1];
    let command_args = &args[2..];

    let env_vars = deserialize_env_vars(serialized_env_vars);

    let status = Command::new(command)
        .envs(env_vars)
        .args(command_args)
        .status()
        .expect(&format!("Failed to execute command {}", command));

    if !status.success() {
        eprintln!(
            "Host {:?} returned non-zero exit code {:?}",
            command,
            status.code()
        );
    }
}
