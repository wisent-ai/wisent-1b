//! The first model: a token stream and a concept stream of vectors.
//!
//! Each layer runs, in order:
//!
//! ```text
//! tokens   ← tokens + CausalSelfAttn(tokens)
//! concepts ← concepts + CrossAttn(concepts → tokens)
//! concepts ← concepts + SelfAttn(concepts)
//! concepts ← concepts + ConceptFFN(concepts)
//! tokens   ← tokens + gate * CrossAttn(tokens → concepts)
//! tokens   ← tokens + TokenFFN(tokens)
//! ```

mod attention;
mod layer;
mod network;
pub(crate) mod parts;

pub use attention::{Attention, AttentionShape};
pub use layer::{ConceptCarry, RejLayer};
pub use network::{Output, Pass, RejRnm};
