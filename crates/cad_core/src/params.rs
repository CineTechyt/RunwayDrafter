//! A parametric pattern: formulas over the avatar's measurements and the
//! user's own variables, guide lines placed by them, and line lengths held
//! to them.
//!
//! * **Formulas** are plain arithmetic in millimetres: numbers, names,
//!   `+ - * /`, brackets and a few functions (`min(a; b)`, `max`, `abs`,
//!   `sqrt`, `round`). Names are the body's measurements (`waist`,
//!   `inseam`, …) and the variables of the document.
//! * **Variables** are named formulas, in any order (a variable may use
//!   one defined below it; a loop is an error).
//! * **Guide lines** (helplines) are horizontal or vertical lines on the
//!   pattern sheet, each at a formula's distance from another guide or
//!   from the sheet's zero. When their formulas give new values (the
//!   avatar changed, say), every piece they cross is stretched between
//!   them: a point on a guide stays on it, a point between two guides
//!   keeps its share of the way between them, and what lies beyond the
//!   outermost guide moves with it.
//! * **Length rules** hold a piece's line at a formula's length.

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::ids::PieceId;

/// A named formula.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub formula: String,
    /// What it is for, shown as a tooltip.
    #[serde(default)]
    pub note: String,
}

/// Which way a guide line runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    /// A line of constant y (the waist line, the hem line).
    Horizontal,
    /// A line of constant x (a centre front, a side).
    Vertical,
}

impl Axis {
    /// The coordinate the guide's position is measured in.
    pub fn across(self, p: Vec2) -> f32 {
        match self {
            Axis::Horizontal => p.y,
            Axis::Vertical => p.x,
        }
    }

    /// The coordinate along the guide.
    pub fn along(self, p: Vec2) -> f32 {
        match self {
            Axis::Horizontal => p.x,
            Axis::Vertical => p.y,
        }
    }

    pub fn other(self) -> Self {
        match self {
            Axis::Horizontal => Axis::Vertical,
            Axis::Vertical => Axis::Horizontal,
        }
    }
}

/// A helpline on the pattern sheet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guide {
    pub id: u32,
    pub name: String,
    pub axis: Axis,
    /// Measured from this guide, or from the sheet's zero.
    pub from: Option<u32>,
    /// Distance from `from` in mm: up or right, or down or left when
    /// `reversed`.
    pub formula: String,
    #[serde(default)]
    pub reversed: bool,
    /// Where it is: y of a horizontal guide, x of a vertical one, mm.
    pub at: f32,
    /// How far it reaches along its direction, mm.
    pub start: f32,
    pub end: f32,
}

impl Guide {
    /// The guide's two end points.
    pub fn ends(&self) -> (Vec2, Vec2) {
        match self.axis {
            Axis::Horizontal => (Vec2::new(self.start, self.at), Vec2::new(self.end, self.at)),
            Axis::Vertical => (Vec2::new(self.at, self.start), Vec2::new(self.at, self.end)),
        }
    }

    /// Whether the guide's reach overlaps `[low, high]` along it.
    pub fn reaches(&self, low: f32, high: f32) -> bool {
        let (a, b) = (self.start.min(self.end), self.start.max(self.end));
        a <= high + 1.0 && b >= low - 1.0
    }
}

/// A line of a piece held at a formula's length.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LengthRule {
    pub piece: PieceId,
    /// Any edge of the line (see `Outline::line_span`).
    pub edge: u32,
    pub formula: String,
}

/// Everything parametric about a document.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Parameters {
    #[serde(default)]
    pub variables: Vec<Variable>,
    #[serde(default)]
    pub guides: Vec<Guide>,
    #[serde(default)]
    pub lengths: Vec<LengthRule>,
    #[serde(default)]
    next_guide: u32,
}

impl Parameters {
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty() && self.guides.is_empty() && self.lengths.is_empty()
    }

    /// Adds a guide at `at` with a plain number for its formula; returns
    /// its id.
    pub fn add_guide(&mut self, axis: Axis, at: f32, start: f32, end: f32) -> u32 {
        self.next_guide = self
            .next_guide
            .max(self.guides.iter().map(|g| g.id).max().unwrap_or(0))
            + 1;
        let id = self.next_guide;
        let count = self.guides.iter().filter(|g| g.axis == axis).count() + 1;
        self.guides.push(Guide {
            id,
            name: match axis {
                Axis::Horizontal => format!("H{count}"),
                Axis::Vertical => format!("V{count}"),
            },
            axis,
            from: None,
            formula: number(at),
            reversed: false,
            at,
            start,
            end,
        });
        id
    }

    pub fn guide(&self, id: u32) -> Option<&Guide> {
        self.guides.iter().find(|g| g.id == id)
    }

    pub fn guide_mut(&mut self, id: u32) -> Option<&mut Guide> {
        self.guides.iter_mut().find(|g| g.id == id)
    }

    /// Removes a guide; guides measured from it stay where they are, at a
    /// plain position.
    pub fn remove_guide(&mut self, id: u32) {
        for guide in &mut self.guides {
            if guide.from == Some(id) {
                guide.from = None;
                guide.formula = number(guide.at);
                guide.reversed = false;
            }
        }
        self.guides.retain(|g| g.id != id);
    }

    /// Measures guide `id` from `from` (or the sheet's zero) by the plain
    /// distance to where it is now, in the direction it lies.
    pub fn measure_guide_from(&mut self, id: u32, from: Option<u32>) {
        let base = from.and_then(|f| self.guide(f)).map_or(0.0, |g| g.at);
        if let Some(guide) = self.guide_mut(id) {
            let distance = guide.at - base;
            guide.from = from;
            guide.reversed = from.is_some() && distance < 0.0;
            guide.formula = number(if from.is_some() {
                distance.abs()
            } else {
                distance
            });
        }
    }

    /// The length rule of the line through `edge` of `piece`, if any.
    pub fn length_rule(&self, piece: PieceId, edges: impl Fn(u32) -> bool) -> Option<&LengthRule> {
        self.lengths
            .iter()
            .find(|r| r.piece == piece && edges(r.edge))
    }
}

/// A number as a formula, without needless decimals.
pub fn number(value: f32) -> String {
    let rounded = (value * 10.0).round() / 10.0;
    if rounded.fract() == 0.0 {
        format!("{rounded:.0}")
    } else {
        format!("{rounded:.1}")
    }
}

/// The plain number a formula is, if it is one.
pub fn as_number(formula: &str) -> Option<f32> {
    let text = formula.trim().replace(',', ".");
    text.parse::<f32>().ok().filter(|v| v.is_finite())
}

/// Why a formula has no value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulaError(pub String);

impl std::fmt::Display for FormulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FormulaError {}

/// Whether `name` can be used as a variable name.
pub fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_')
        && !FUNCTIONS.contains(&name)
}

const FUNCTIONS: [&str; 5] = ["min", "max", "abs", "sqrt", "round"];

/// Every name a formula uses (functions left out).
pub fn names_in(formula: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut lexer = Lexer::new(formula);
    while let Ok(Some(token)) = lexer.next() {
        if let Token::Name(name) = token
            && !FUNCTIONS.contains(&name.as_str())
            && !names.contains(&name)
        {
            names.push(name);
        }
    }
    names
}

/// Evaluates a formula; `lookup` gives the value of a name.
pub fn evaluate(formula: &str, lookup: &dyn Fn(&str) -> Option<f32>) -> Result<f32, FormulaError> {
    let mut parser = Parser {
        lexer: Lexer::new(formula),
        peeked: None,
        lookup,
    };
    if parser.peek()?.is_none() {
        return Err(FormulaError("empty".into()));
    }
    let value = parser.expression()?;
    match parser.peek()? {
        None => {}
        Some(token) => return Err(FormulaError(format!("unexpected {}", token.describe()))),
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FormulaError(
            "not a finite number (division by zero?)".into(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f32),
    Name(String),
    Symbol(char),
}

impl Token {
    fn describe(&self) -> String {
        match self {
            Token::Number(v) => format!("number {v}"),
            Token::Name(n) => format!("“{n}”"),
            Token::Symbol(c) => format!("“{c}”"),
        }
    }
}

struct Lexer<'a> {
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    text: &'a str,
}

impl<'a> Lexer<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            chars: text.char_indices().peekable(),
            text,
        }
    }

    fn next(&mut self) -> Result<Option<Token>, FormulaError> {
        while self.chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        let Some((start, c)) = self.chars.next() else {
            return Ok(None);
        };
        if c.is_ascii_digit() || c == '.' {
            let mut end = start + c.len_utf8();
            while let Some(&(i, d)) = self.chars.peek() {
                // A comma is a decimal comma only between digits.
                let decimal_comma = d == ','
                    && self.text[i + 1..]
                        .chars()
                        .next()
                        .is_some_and(|n| n.is_ascii_digit())
                    && self.text[start..i].chars().all(|c| c.is_ascii_digit());
                if d.is_ascii_digit() || d == '.' || decimal_comma {
                    end = i + d.len_utf8();
                    self.chars.next();
                } else {
                    break;
                }
            }
            let text = self.text[start..end].replace(',', ".");
            return text
                .parse::<f32>()
                .map(|v| Some(Token::Number(v)))
                .map_err(|_| FormulaError(format!("“{text}” is not a number")));
        }
        if c.is_alphabetic() || c == '_' {
            let mut end = start + c.len_utf8();
            while let Some(&(i, d)) = self.chars.peek() {
                if d.is_alphanumeric() || d == '_' {
                    end = i + d.len_utf8();
                    self.chars.next();
                } else {
                    break;
                }
            }
            return Ok(Some(Token::Name(self.text[start..end].to_owned())));
        }
        if "+-*/();,".contains(c) {
            return Ok(Some(Token::Symbol(c)));
        }
        Err(FormulaError(format!("“{c}” cannot be used in a formula")))
    }
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    peeked: Option<Option<Token>>,
    lookup: &'a dyn Fn(&str) -> Option<f32>,
}

impl Parser<'_> {
    fn peek(&mut self) -> Result<Option<Token>, FormulaError> {
        if self.peeked.is_none() {
            self.peeked = Some(self.lexer.next()?);
        }
        Ok(self.peeked.clone().flatten())
    }

    fn take(&mut self) -> Result<Option<Token>, FormulaError> {
        let token = self.peek()?;
        self.peeked = None;
        Ok(token)
    }

    fn expression(&mut self) -> Result<f32, FormulaError> {
        let mut value = self.term()?;
        while let Some(Token::Symbol(op @ ('+' | '-'))) = self.peek()? {
            self.take()?;
            let rhs = self.term()?;
            value = if op == '+' { value + rhs } else { value - rhs };
        }
        Ok(value)
    }

    fn term(&mut self) -> Result<f32, FormulaError> {
        let mut value = self.factor()?;
        while let Some(Token::Symbol(op @ ('*' | '/'))) = self.peek()? {
            self.take()?;
            let rhs = self.factor()?;
            value = if op == '*' { value * rhs } else { value / rhs };
        }
        Ok(value)
    }

    fn factor(&mut self) -> Result<f32, FormulaError> {
        match self.take()? {
            Some(Token::Number(v)) => Ok(v),
            Some(Token::Symbol('-')) => Ok(-self.factor()?),
            Some(Token::Symbol('+')) => self.factor(),
            Some(Token::Symbol('(')) => {
                let value = self.expression()?;
                self.expect(')')?;
                Ok(value)
            }
            Some(Token::Name(name)) => {
                if FUNCTIONS.contains(&name.as_str()) {
                    self.expect('(')?;
                    let mut args = vec![self.expression()?];
                    while let Some(Token::Symbol(';' | ',')) = self.peek()? {
                        self.take()?;
                        args.push(self.expression()?);
                    }
                    self.expect(')')?;
                    return call(&name, &args);
                }
                (self.lookup)(&name).ok_or_else(|| FormulaError(format!("unknown name “{name}”")))
            }
            Some(token) => Err(FormulaError(format!("unexpected {}", token.describe()))),
            None => Err(FormulaError("the formula ends too early".into())),
        }
    }

    fn expect(&mut self, symbol: char) -> Result<(), FormulaError> {
        match self.take()? {
            Some(Token::Symbol(c)) if c == symbol => Ok(()),
            Some(token) => Err(FormulaError(format!(
                "expected “{symbol}”, found {}",
                token.describe()
            ))),
            None => Err(FormulaError(format!("missing “{symbol}”"))),
        }
    }
}

fn call(name: &str, args: &[f32]) -> Result<f32, FormulaError> {
    let one = || {
        if args.len() == 1 {
            Ok(args[0])
        } else {
            Err(FormulaError(format!("{name}() takes one value")))
        }
    };
    match name {
        "min" => Ok(args.iter().copied().fold(f32::INFINITY, f32::min)),
        "max" => Ok(args.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
        "abs" => one().map(f32::abs),
        "sqrt" => one().map(f32::sqrt),
        "round" => one().map(f32::round),
        _ => Err(FormulaError(format!("unknown function {name}()"))),
    }
}

/// What a name stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    Measurement,
    Variable,
    Guide,
}

/// One name with its value (or why it has none).
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    pub value: Result<f32, FormulaError>,
    /// The formula (variables), or what it measures (measurements).
    pub detail: String,
}

/// A measurement of the avatar for formulas: its name, value in mm and
/// what it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub name: String,
    pub value: f32,
    pub label: String,
}

/// Evaluates every variable against the measurements. Variables may use
/// each other in any order; loops and unknown names are errors.
pub fn resolve(parameters: &Parameters, measurements: &[Measured]) -> Vec<Param> {
    let mut known: HashMap<String, f32> = measurements
        .iter()
        .map(|m| (m.name.clone(), m.value))
        .collect();
    let mut results: HashMap<String, Result<f32, FormulaError>> = HashMap::new();
    fn solve(
        name: &str,
        parameters: &Parameters,
        known: &mut HashMap<String, f32>,
        results: &mut HashMap<String, Result<f32, FormulaError>>,
        visiting: &mut Vec<String>,
    ) -> Option<f32> {
        if let Some(value) = known.get(name) {
            return Some(*value);
        }
        if let Some(result) = results.get(name) {
            return result.as_ref().ok().copied();
        }
        let variable = parameters.variables.iter().find(|v| v.name == name)?;
        if visiting.iter().any(|v| v == name) {
            results.insert(
                name.to_owned(),
                Err(FormulaError(format!("“{name}” depends on itself"))),
            );
            return None;
        }
        visiting.push(name.to_owned());
        for used in names_in(&variable.formula) {
            solve(&used, parameters, known, results, visiting);
        }
        visiting.pop();
        let result = evaluate(&variable.formula, &|n| {
            known
                .get(n)
                .copied()
                .or_else(|| results.get(n)?.as_ref().ok().copied())
        });
        let result = match (result, results.get(name)) {
            // Found to loop while its names were solved.
            (_, Some(Err(error))) => Err(error.clone()),
            (result, _) => result,
        };
        if let Ok(value) = result {
            known.insert(name.to_owned(), value);
        }
        results.insert(name.to_owned(), result.clone());
        result.ok()
    }
    let mut params: Vec<Param> = measurements
        .iter()
        .map(|m| Param {
            name: m.name.clone(),
            kind: ParamKind::Measurement,
            value: Ok(m.value),
            detail: m.label.clone(),
        })
        .collect();
    for variable in &parameters.variables {
        solve(
            &variable.name,
            parameters,
            &mut known,
            &mut results,
            &mut Vec::new(),
        );
        let value = if measurements.iter().any(|m| m.name == variable.name) {
            Err(FormulaError(format!(
                "“{}” is a measurement's name",
                variable.name
            )))
        } else if !valid_name(&variable.name) {
            Err(FormulaError(
                "names are letters, digits and _, starting with a letter".into(),
            ))
        } else {
            results
                .get(&variable.name)
                .cloned()
                .unwrap_or_else(|| Err(FormulaError("no value".into())))
        };
        params.push(Param {
            name: variable.name.clone(),
            kind: ParamKind::Variable,
            value,
            detail: variable.formula.clone(),
        });
    }
    params
}

/// Looks a name up among resolved parameters.
pub fn lookup(params: &[Param]) -> impl Fn(&str) -> Option<f32> + '_ {
    move |name| {
        params
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| p.value.as_ref().ok().copied())
    }
}

/// Where every guide belongs by its formula, by id. Guides whose formula
/// has no value (or that are measured from themselves) are left out.
pub fn guide_positions(
    parameters: &Parameters,
    params: &[Param],
) -> HashMap<u32, Result<f32, FormulaError>> {
    let names = lookup(params);
    let mut out: HashMap<u32, Result<f32, FormulaError>> = HashMap::new();
    fn place(
        id: u32,
        parameters: &Parameters,
        names: &dyn Fn(&str) -> Option<f32>,
        out: &mut HashMap<u32, Result<f32, FormulaError>>,
        depth: usize,
    ) -> Result<f32, FormulaError> {
        if let Some(done) = out.get(&id) {
            return done.clone();
        }
        let Some(guide) = parameters.guide(id) else {
            return Err(FormulaError("measured from a guide that is gone".into()));
        };
        if depth > parameters.guides.len() {
            return Err(FormulaError(
                "guides measured from each other in a loop".into(),
            ));
        }
        let base = match guide.from {
            Some(from) => place(from, parameters, names, out, depth + 1)
                .map_err(|e| FormulaError(format!("from: {e}")))?,
            None => 0.0,
        };
        let sign = if guide.reversed { -1.0 } else { 1.0 };
        let result = evaluate(&guide.formula, names).map(|d| base + sign * d);
        out.insert(id, result.clone());
        result
    }
    for guide in &parameters.guides {
        let _ = place(guide.id, parameters, &names, &mut out, 0);
    }
    out
}

/// A piecewise linear map of one coordinate: `(old, new)` pairs sorted by
/// `old`. Between two pairs it interpolates; beyond the ends it moves with
/// the nearest pair.
#[derive(Debug, Clone, Default)]
pub struct Stretch {
    pairs: Vec<(f32, f32)>,
}

impl Stretch {
    pub fn new(mut pairs: Vec<(f32, f32)>) -> Self {
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Guides on top of each other: keep one.
        pairs.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-3);
        Self { pairs }
    }

    /// Whether the map moves anything.
    pub fn moves(&self) -> bool {
        self.pairs.iter().any(|(old, new)| (old - new).abs() > 1e-4)
    }

    pub fn map(&self, x: f32) -> f32 {
        let pairs = &self.pairs;
        match pairs.len() {
            0 => x,
            1 => x + pairs[0].1 - pairs[0].0,
            n => {
                if x <= pairs[0].0 {
                    return x + pairs[0].1 - pairs[0].0;
                }
                if x >= pairs[n - 1].0 {
                    return x + pairs[n - 1].1 - pairs[n - 1].0;
                }
                let k = pairs.partition_point(|p| p.0 <= x).clamp(1, n - 1);
                let ((a0, a1), (b0, b1)) = (pairs[k - 1], pairs[k]);
                let t = (x - a0) / (b0 - a0);
                a1 + t * (b1 - a1)
            }
        }
    }
}

/// What applying the parameters did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ApplyReport {
    /// Pieces whose shape changed.
    pub pieces: Vec<PieceId>,
    /// Guides that moved.
    pub guides_moved: bool,
    /// Formulas that have no value, with what they belong to.
    pub errors: Vec<(String, FormulaError)>,
}

impl ApplyReport {
    pub fn changed(&self) -> bool {
        self.guides_moved || !self.pieces.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(name: &str) -> Option<f32> {
        match name {
            "waist" => Some(700.0),
            "inseam" => Some(780.0),
            _ => None,
        }
    }

    #[test]
    fn formulas_are_evaluated_with_the_usual_precedence() {
        assert_eq!(evaluate("waist / 4 + 10", &names), Ok(185.0));
        assert_eq!(evaluate("-(inseam - 30) * 2", &names), Ok(-1500.0));
        assert_eq!(
            evaluate("max(waist; inseam) - min(1, 2)", &names),
            Ok(779.0)
        );
        assert_eq!(evaluate("12,5 + 0.5", &names), Ok(13.0));
        assert!(evaluate("waist +", &names).is_err());
        assert!(evaluate("hips", &names).unwrap_err().0.contains("hips"));
        assert!(evaluate("1 / 0", &names).is_err());
        assert_eq!(names_in("max(waist; ease) + waist"), ["waist", "ease"]);
    }

    #[test]
    fn variables_use_each_other_in_any_order_and_loops_are_errors() {
        let measured = [Measured {
            name: "waist".into(),
            value: 700.0,
            label: String::new(),
        }];
        let variable = |name: &str, formula: &str| Variable {
            name: name.into(),
            formula: formula.into(),
            note: String::new(),
        };
        let parameters = Parameters {
            variables: vec![
                variable("front", "quarter + ease"),
                variable("quarter", "waist / 4"),
                variable("ease", "10"),
                variable("a", "b + 1"),
                variable("b", "a + 1"),
            ],
            ..Parameters::default()
        };
        let params = resolve(&parameters, &measured);
        let value = |n: &str| params.iter().find(|p| p.name == n).unwrap().value.clone();
        assert_eq!(value("front"), Ok(185.0));
        assert!(value("a").is_err());
        assert!(value("b").is_err());
    }

    #[test]
    fn a_stretch_interpolates_between_guides_and_moves_beyond_them() {
        let s = Stretch::new(vec![(0.0, 0.0), (-800.0, -900.0)]);
        assert_eq!(s.map(0.0), 0.0);
        assert_eq!(s.map(-800.0), -900.0);
        assert_eq!(s.map(-400.0), -450.0);
        assert_eq!(s.map(50.0), 50.0);
        assert_eq!(s.map(-850.0), -950.0);
    }

    /// The trouser example: a waist line, a hem line `inseam` below it, and
    /// a leg drawn between them. A longer inseam stretches the leg; the
    /// waistband above the waist line just moves with it; a line held at
    /// `waist / 4` follows the waist.
    #[test]
    fn a_shape_drawn_on_guides_stretches_with_the_measurements() {
        use crate::{Outline, PatternDocument};
        let mut doc = PatternDocument::new();
        let fabric = doc.add_fabric("Denim", [0.1, 0.1, 0.3]);
        let leg = doc.add_piece(
            "Leg",
            Outline::polygon([
                Vec2::new(0.0, -780.0),
                Vec2::new(200.0, -780.0),
                Vec2::new(200.0, 0.0),
                Vec2::new(220.0, 40.0),
                Vec2::new(0.0, 40.0),
            ]),
            fabric,
        );
        let parameters = doc.parameters_mut();
        let waist = parameters.add_guide(Axis::Horizontal, 0.0, -50.0, 400.0);
        let hem = parameters.add_guide(Axis::Horizontal, -780.0, -50.0, 400.0);
        let guide = parameters.guide_mut(hem).unwrap();
        guide.from = Some(waist);
        guide.formula = "-inseam".into();
        parameters.lengths.push(LengthRule {
            piece: leg,
            edge: 0,
            formula: "waist / 4".into(),
        });
        let measured = |inseam: f32| {
            vec![
                Measured {
                    name: "inseam".into(),
                    value: inseam,
                    label: String::new(),
                },
                Measured {
                    name: "waist".into(),
                    value: 800.0,
                    label: String::new(),
                },
            ]
        };
        let report = doc.apply_parameters(&measured(780.0));
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert!(
            !report.guides_moved,
            "the guides already are where they belong"
        );

        let report = doc.apply_parameters(&measured(830.0));
        assert!(report.guides_moved && report.pieces.contains(&leg));
        let ys: Vec<f32> = doc
            .piece(leg)
            .unwrap()
            .outline
            .points()
            .iter()
            .map(|p| p.position.y)
            .collect();
        assert!(
            (ys[0] + 830.0).abs() < 1e-3,
            "the hem is on the hem line: {ys:?}"
        );
        assert!(
            ys[2].abs() < 1e-3 && (ys[3] - 40.0).abs() < 1e-3,
            "the waist stays: {ys:?}"
        );
        let outline = &doc.piece(leg).unwrap().outline;
        let hem_length = outline.span_length(outline.line_span(0));
        assert!(
            (hem_length - 200.0).abs() < 0.1,
            "held at waist / 4: {hem_length}"
        );
        assert_eq!(doc.parameters().guide(hem).unwrap().at, -830.0);
    }
}
