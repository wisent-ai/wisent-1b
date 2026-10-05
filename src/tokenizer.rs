//! Text to token ids and back.
//!
//! The native tokenizer is character-level: `<PAD>`, `<EOS>`, `<UNK>`, then
//! printable ASCII, newline and tab, then the printable Latin-1 supplement,
//! cut to the model's vocabulary. A pretrained tokenizer is read from a
//! Hugging Face `tokenizer.json`; one with more ids than the model has
//! embeddings is refused instead of having its ids clipped.

use std::collections::HashMap;
use std::path::Path;

use crate::Error;

/// U+0020 SPACE to U+007E TILDE: printable ASCII.
const PRINTABLE_ASCII: std::ops::Range<u32> = 0x20..0x7F;
/// U+00A1 to U+00FF: the printable Latin-1 supplement.
const PRINTABLE_LATIN1: std::ops::Range<u32> = 0xA1..0x100;
const SPECIAL_TOKENS: [&str; 3] = ["<PAD>", "<EOS>", "<UNK>"];
const PAD: u32 = 0;
const EOS: u32 = 1;
const UNK: u32 = 2;

#[derive(Clone, Debug)]
pub enum RejTokenizer {
    Native {
        symbols: Vec<String>,
        ids: HashMap<char, u32>,
    },
    Pretrained {
        tokenizer: Box<tokenizers::Tokenizer>,
        eos: Option<u32>,
        pad: Option<u32>,
    },
}

impl RejTokenizer {
    /// The character tokenizer, cut to `vocab_size` symbols.
    pub fn native(vocab_size: usize) -> Self {
        let characters = PRINTABLE_ASCII
            .chain(['\n' as u32, '\t' as u32])
            .chain(PRINTABLE_LATIN1)
            .filter_map(char::from_u32)
            .map(String::from);
        let symbols: Vec<String> = SPECIAL_TOKENS
            .iter()
            .map(|token| token.to_string())
            .chain(characters)
            .take(vocab_size)
            .collect();
        let ids = symbols
            .iter()
            .enumerate()
            .skip(SPECIAL_TOKENS.len())
            .filter_map(|(id, symbol)| symbol.chars().next().map(|ch| (ch, id as u32)))
            .collect();
        Self::Native { symbols, ids }
    }

    /// A Hugging Face tokenizer file. `eos_token` and `pad_token` name the
    /// tokens that end generation and pad a batch, when the run uses them.
    pub fn pretrained(
        path: &Path,
        vocab_size: usize,
        eos_token: Option<&str>,
        pad_token: Option<&str>,
    ) -> Result<Self, Error> {
        let refuse = |message: String| Error::Tokenizer { path: path.to_path_buf(), message };
        let tokenizer = tokenizers::Tokenizer::from_file(path).map_err(|e| refuse(e.to_string()))?;
        let size = tokenizer.get_vocab_size(true);
        if size > vocab_size {
            return Err(refuse(format!("has {size} ids but the model has {vocab_size} embeddings")));
        }
        let id = |token: Option<&str>| -> Result<Option<u32>, Error> {
            token
                .map(|name| tokenizer.token_to_id(name).ok_or_else(|| refuse(format!("has no token {name}"))))
                .transpose()
        };
        let (eos, pad) = (id(eos_token)?, id(pad_token)?);
        Ok(Self::Pretrained { tokenizer: Box::new(tokenizer), eos, pad })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        match self {
            Self::Native { ids, .. } => Ok(text
                .chars()
                .map(|ch| ids.get(&ch).copied().unwrap_or(UNK))
                .collect()),
            Self::Pretrained { tokenizer, .. } => tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|e| Error::Control(format!("tokenizer could not encode the text: {e}"))),
        }
    }

    /// The text of `ids`, without special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        match self {
            Self::Native { symbols, .. } => Ok(ids
                .iter()
                .filter(|id| ![PAD, EOS, UNK].contains(id))
                .map(|&id| symbols.get(id as usize).map_or("<?>", String::as_str))
                .collect()),
            Self::Pretrained { tokenizer, .. } => tokenizer
                .decode(ids, true)
                .map_err(|e| Error::Control(format!("tokenizer could not decode ids: {e}"))),
        }
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        match self {
            Self::Native { .. } => Some(EOS),
            Self::Pretrained { eos, .. } => *eos,
        }
    }

    pub fn pad_token_id(&self) -> Option<u32> {
        match self {
            Self::Native { .. } => Some(PAD),
            Self::Pretrained { pad, .. } => *pad,
        }
    }
}
