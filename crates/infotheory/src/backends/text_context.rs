use ahash::AHashMap;

type RepeatPos = u32;
const REPEAT_POS_NONE: RepeatPos = u32::MAX;
type RepeatKey = u32;

#[inline]
fn repeat_pos_from_usize(pos: usize) -> RepeatPos {
    if pos >= REPEAT_POS_NONE as usize {
        panic!("text repeat position overflow");
    }
    pos as RepeatPos
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NeuralContextState {
    pub(crate) prev1: u8,
    pub(crate) prev2: u8,
    pub(crate) prev1_class: u8,
    pub(crate) prev2_class: u8,
    pub(crate) run_len: u16,
    pub(crate) utf8_left: u8,
    pub(crate) in_word: bool,
    pub(crate) word_len_bucket: u8,
    pub(crate) prev_word_class: u8,
    pub(crate) bracket_bucket: u8,
    pub(crate) quote_flags: u8,
    pub(crate) sentence_boundary: bool,
    pub(crate) paragraph_break: bool,
    pub(crate) repeat_len_bucket: u8,
    pub(crate) copied_last_byte: bool,
    pub(crate) has_history: bool,
}

pub(crate) type NeuralHistoryState = NeuralContextState;

#[derive(Clone, Copy, Debug, Default)]
struct WordTracker {
    len: u16,
    saw_lower: bool,
    saw_upper: bool,
    saw_digit: bool,
    saw_other: bool,
}

impl WordTracker {
    fn clear(&mut self) {
        *self = Self::default();
    }

    fn observe(&mut self, byte: u8) {
        self.len = self.len.saturating_add(1);
        match byte {
            b'a'..=b'z' => self.saw_lower = true,
            b'A'..=b'Z' => self.saw_upper = true,
            b'0'..=b'9' => self.saw_digit = true,
            _ => self.saw_other = true,
        }
    }

    fn class(self) -> u8 {
        if self.saw_digit && !self.saw_lower && !self.saw_upper && !self.saw_other {
            3
        } else if self.saw_upper && !self.saw_lower && !self.saw_digit && !self.saw_other {
            2
        } else if self.saw_lower && !self.saw_upper && !self.saw_digit && !self.saw_other {
            1
        } else if self.len > 0 {
            4
        } else {
            0
        }
    }
}

#[derive(Clone, Debug, Default)]
struct LocalRepeatState {
    history: Vec<u8>,
    table: AHashMap<RepeatKey, (RepeatPos, RepeatPos)>,
    predicted: Option<u8>,
    match_len: usize,
    copied_last_byte: bool,
}

impl LocalRepeatState {
    const MIN_LEN: usize = 4;
    const MAX_LEN: usize = 255;

    fn predict_from_history(&self) -> (Option<u8>, usize) {
        if self.history.len() < Self::MIN_LEN {
            return (None, 0);
        }
        let end = self.history.len() - 1;
        let Some(key) = repeat_key(&self.history) else {
            return (None, 0);
        };
        let Some(&(latest, previous)) = self.table.get(&key) else {
            return (None, 0);
        };
        let end_pos = repeat_pos_from_usize(end);
        let candidate_end = if latest == end_pos { previous } else { latest };
        if candidate_end == REPEAT_POS_NONE {
            return (None, 0);
        }
        let candidate_end = candidate_end as usize;
        if candidate_end + 1 >= self.history.len() {
            return (None, 0);
        }
        let mut matched = Self::MIN_LEN;
        while matched < Self::MAX_LEN
            && end >= matched
            && candidate_end >= matched
            && self.history[end - matched] == self.history[candidate_end - matched]
        {
            matched += 1;
        }
        (Some(self.history[candidate_end + 1]), matched)
    }

    fn repeat_len_bucket(&self) -> u8 {
        bucket_repeat_len(self.match_len)
    }

    fn update(&mut self, symbol: u8) {
        self.copied_last_byte = self.predicted == Some(symbol);
        self.history.push(symbol);
        if let Some(key) = repeat_key(&self.history) {
            let end = repeat_pos_from_usize(self.history.len() - 1);
            self.table
                .entry(key)
                .and_modify(|entry| {
                    entry.1 = entry.0;
                    entry.0 = end;
                })
                .or_insert((end, REPEAT_POS_NONE));
        }
        let (predicted, match_len) = self.predict_from_history();
        self.predicted = predicted;
        self.match_len = match_len;
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TextContextAnalyzer {
    state: NeuralContextState,
    word: WordTracker,
    newline_run: u8,
    bracket_stack: [u8; 8],
    bracket_depth: usize,
    repeat: LocalRepeatState,
    track_repeats: bool,
}

impl TextContextAnalyzer {
    pub(crate) fn new() -> Self {
        Self {
            track_repeats: true,
            ..Self::default()
        }
    }

    /// Construct an analyzer that omits local-copy tracking.
    ///
    /// This is used by predictors that consume only lexical/structural fields;
    /// it avoids maintaining a redundant match table and keeps checkpoints
    /// constant-size in the length of the observed stream.
    #[cfg(feature = "backend-context")]
    pub(crate) fn without_repeat_tracking() -> Self {
        Self::default()
    }

    pub(crate) fn state(&self) -> NeuralContextState {
        self.state
    }

    pub(crate) fn update(&mut self, symbol: u8) {
        let was_predicted: Option<u8> = if self.track_repeats {
            let predicted: Option<u8> = self.repeat.predicted;
            self.repeat.update(symbol);
            predicted
        } else {
            None
        };

        let byte_class = classify_byte(symbol);
        if self.state.has_history && symbol == self.state.prev1 {
            self.state.run_len = self.state.run_len.saturating_add(1).min(255);
        } else {
            self.state.run_len = 1;
        }

        let prev_in_word = self.state.in_word;
        let is_word_byte = is_word_byte(symbol);
        if is_word_byte {
            if !prev_in_word {
                self.word.clear();
            }
            self.word.observe(symbol);
        } else if prev_in_word {
            self.state.prev_word_class = self.word.class();
            self.word.clear();
        }
        self.state.in_word = is_word_byte;
        self.state.word_len_bucket = bucket_word_len(self.word.len);

        self.update_structure(symbol);

        self.state.prev2 = self.state.prev1;
        self.state.prev2_class = self.state.prev1_class;
        self.state.prev1 = symbol;
        self.state.prev1_class = byte_class;
        self.state.utf8_left = utf8_left_after(symbol, self.state.utf8_left);
        self.state.repeat_len_bucket = if self.track_repeats {
            self.repeat.repeat_len_bucket()
        } else {
            0
        };
        self.state.copied_last_byte = was_predicted == Some(symbol);
        self.state.has_history = true;
    }

    fn update_structure(&mut self, symbol: u8) {
        self.state.sentence_boundary = matches!(symbol, b'.' | b'!' | b'?');

        if symbol == b'\n' {
            self.newline_run = self.newline_run.saturating_add(1).min(3);
        } else {
            self.newline_run = 0;
        }
        self.state.paragraph_break = self.newline_run >= 2;

        match symbol {
            b'(' => self.push_bracket(1),
            b'[' => self.push_bracket(2),
            b'{' => self.push_bracket(3),
            b'<' => self.push_bracket(4),
            b')' => self.pop_bracket(1),
            b']' => self.pop_bracket(2),
            b'}' => self.pop_bracket(3),
            b'>' => self.pop_bracket(4),
            b'"' => self.state.quote_flags ^= 0x1,
            b'\'' => self.state.quote_flags ^= 0x2,
            _ => {}
        }
        self.state.bracket_bucket = if self.bracket_depth == 0 {
            0
        } else {
            self.bracket_stack[self.bracket_depth - 1]
        };
    }

    fn push_bracket(&mut self, bracket: u8) {
        if self.bracket_depth < self.bracket_stack.len() {
            self.bracket_stack[self.bracket_depth] = bracket;
            self.bracket_depth += 1;
        } else {
            self.bracket_stack[self.bracket_stack.len() - 1] = bracket;
        }
    }

    fn pop_bracket(&mut self, bracket: u8) {
        if self.bracket_depth == 0 {
            return;
        }
        if self.bracket_stack[self.bracket_depth - 1] == bracket {
            self.bracket_depth -= 1;
            return;
        }
        for idx in (0..self.bracket_depth).rev() {
            if self.bracket_stack[idx] == bracket {
                self.bracket_depth = idx;
                return;
            }
        }
    }
}

#[inline]
pub(crate) fn classify_byte(byte: u8) -> u8 {
    match byte {
        b'a'..=b'z' | b'A'..=b'Z' => 1,
        b'0'..=b'9' => 2,
        b' ' | b'\t' | b'\n' | b'\r' => 3,
        b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~' => 4,
        0xC0..=0xFF => 5,
        0x80..=0xBF => 6,
        _ => 0,
    }
}

#[inline]
pub(crate) fn is_word_byte(byte: u8) -> bool {
    matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | 0x80..=0xFF)
}

#[inline]
pub(crate) fn bucket_word_len(len: u16) -> u8 {
    match len {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        _ => 6,
    }
}

#[inline]
pub(crate) fn bucket_repeat_len(len: usize) -> u8 {
    match len {
        0 => 0,
        1..=3 => 1,
        4..=5 => 2,
        6..=8 => 3,
        9..=12 => 4,
        13..=16 => 5,
        17..=24 => 6,
        _ => 7,
    }
}

#[inline]
fn utf8_left_after(symbol: u8, prev_left: u8) -> u8 {
    if (0x80..=0xBF).contains(&symbol) {
        prev_left.saturating_sub(1)
    } else {
        match symbol {
            0xC0..=0xDF => 1,
            0xE0..=0xEF => 2,
            0xF0..=0xF7 => 3,
            _ => 0,
        }
    }
}

#[inline]
fn repeat_key(history: &[u8]) -> Option<RepeatKey> {
    if history.len() < LocalRepeatState::MIN_LEN {
        return None;
    }
    let n = history.len();
    Some(u32::from_be_bytes([
        history[n - 4],
        history[n - 3],
        history[n - 2],
        history[n - 1],
    ]))
}
