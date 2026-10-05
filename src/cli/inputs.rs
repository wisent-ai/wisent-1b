//! Reading what the command line is given: `name=value` controls, corpora and
//! labelled JSON Lines files.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use rej_1b::Error;

/// `name=value`.
pub fn named_value(text: &str) -> Result<(String, f64), String> {
    let (name, value) = split(text)?;
    let value = value.parse::<f64>().map_err(|e| format!("'{text}': {e}"))?;
    Ok((name, value))
}

/// `name=v1,v2,...`.
pub fn named_vector(text: &str) -> Result<(String, Vec<f64>), String> {
    let (name, values) = split(text)?;
    let values = values
        .split(',')
        .map(|value| value.trim().parse::<f64>().map_err(|e| format!("'{text}': {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((name, values))
}

fn split(text: &str) -> Result<(String, &str), String> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| format!("'{text}' is not name=value"))?;
    Ok((name.trim().to_string(), value.trim()))
}

pub fn read_text(path: &Path) -> Result<String, Error> {
    std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })
}

/// One line of a labelled training file: a text, optionally its translation,
/// and the magnitude of each named concept it shows. A concept the line does
/// not name has magnitude zero.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelledLine {
    pub text: String,
    #[serde(default)]
    pub parallel: Option<String>,
    pub controls: BTreeMap<String, f64>,
}

pub fn read_labelled(path: &Path) -> Result<Vec<LabelledLine>, Error> {
    read_text(path)?
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .map_err(|e| Error::Training(format!("{} line {}: {e}", path.display(), index + 1)))
        })
        .collect()
}
