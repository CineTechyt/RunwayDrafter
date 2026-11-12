//! FBX: avatar import (binary and ASCII, version 7.x) and garment export
//! (7.4, binary or ASCII).
//!
//! FBX is a tree of named nodes with a property list each; the binary and
//! the text encoding carry the same tree. Both are parsed into [`Node`]s,
//! and the geometry is then read from the tree:
//!
//! * `Objects/Geometry` nodes hold `Vertices` and `PolygonVertexIndex`
//!   (the last corner of every polygon is stored as `-(index + 1)`);
//! * `Objects/Model` nodes hold the transforms, and `Connections` say
//!   which model a geometry hangs on and what that model's parent is;
//! * `GlobalSettings` holds the unit (centimetres per unit) and up axis.
//!
//! Not supported: pivots and geometric offsets of models, skinning (meshes
//! are read in their bind pose), and FBX 6.x files.

use std::collections::HashMap;
use std::fmt::Write as _;

use glam::{DMat4, DVec3, Vec3};

use crate::IoError;
use crate::mesh::ImportedMesh;
use crate::obj::GarmentExport;

const BINARY_MAGIC: &[u8] = b"Kaydara FBX Binary  \0";

#[derive(Debug, Clone, PartialEq)]
pub enum Property {
    Integer(i64),
    Float(f64),
    Text(String),
    Integers(Vec<i64>),
    Floats(Vec<f64>),
    /// Binary blobs and boolean arrays; nothing geometry needs.
    Other,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Node {
    pub name: String,
    pub properties: Vec<Property>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }

    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }

    /// Every number in the property list, arrays flattened.
    pub fn numbers(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for property in &self.properties {
            match property {
                Property::Integer(i) => out.push(*i as f64),
                Property::Float(f) => out.push(*f),
                Property::Integers(list) => out.extend(list.iter().map(|&i| i as f64)),
                Property::Floats(list) => out.extend(list),
                Property::Text(_) | Property::Other => {}
            }
        }
        out
    }

    fn integer(&self, index: usize) -> Option<i64> {
        match self.properties.get(index)? {
            Property::Integer(i) => Some(*i),
            Property::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    fn text(&self, index: usize) -> Option<&str> {
        match self.properties.get(index)? {
            Property::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The values of `P: "name", …` under this node's `Properties70`.
    fn property70(&self, name: &str) -> Option<Vec<f64>> {
        self.child("Properties70")?.children_named("P").find(|p| p.text(0) == Some(name)).map(Node::numbers)
    }
}

/// Parses a binary or ASCII FBX file into its node tree; the returned
/// node is an unnamed root.
pub fn parse_fbx(bytes: &[u8]) -> Result<Node, IoError> {
    if bytes.starts_with(BINARY_MAGIC) { parse_binary(bytes) } else { parse_ascii(bytes) }
}

// ---------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], IoError> {
        let slice = self.data.get(self.at..self.at + count).ok_or_else(|| IoError::malformed("unexpected end of FBX data"))?;
        self.at += count;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], IoError> {
        Ok(self.take(N)?.try_into().expect("length checked"))
    }

    fn u32(&mut self) -> Result<u32, IoError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    /// A record offset or count: 32 bits before FBX 7.5, 64 bits since.
    fn size(&mut self, wide: bool) -> Result<u64, IoError> {
        Ok(if wide { u64::from_le_bytes(self.array()?) } else { self.u32()? as u64 })
    }
}

fn parse_binary(bytes: &[u8]) -> Result<Node, IoError> {
    let mut reader = Reader { data: bytes, at: BINARY_MAGIC.len() + 2 };
    let version = reader.u32()?;
    let wide = version >= 7500;
    let mut root = Node::default();
    while reader.at < bytes.len() {
        match binary_node(&mut reader, wide, 0)? {
            Some(node) => root.children.push(node),
            None => break, // the footer follows
        }
    }
    Ok(root)
}

/// Reads one node record; `None` for the all-zero record that ends a list.
fn binary_node(reader: &mut Reader<'_>, wide: bool, depth: u32) -> Result<Option<Node>, IoError> {
    if depth > 64 {
        return Err(IoError::malformed("FBX nodes nested too deeply"));
    }
    let end = reader.size(wide)? as usize;
    let property_count = reader.size(wide)?;
    let _property_bytes = reader.size(wide)?;
    let name_length = reader.take(1)?[0] as usize;
    let name = String::from_utf8_lossy(reader.take(name_length)?).into_owned();
    if end == 0 {
        return Ok(None);
    }
    if end > reader.data.len() || end < reader.at {
        return Err(IoError::malformed("FBX node runs past the file"));
    }
    let mut node = Node { name, ..Node::default() };
    for _ in 0..property_count {
        node.properties.push(binary_property(reader)?);
    }
    while reader.at < end {
        match binary_node(reader, wide, depth + 1)? {
            Some(child) => node.children.push(child),
            None => break,
        }
    }
    reader.at = end;
    Ok(Some(node))
}

fn binary_property(reader: &mut Reader<'_>) -> Result<Property, IoError> {
    let kind = reader.take(1)?[0];
    Ok(match kind {
        b'Y' => Property::Integer(i16::from_le_bytes(reader.array()?) as i64),
        b'C' => Property::Integer(reader.take(1)?[0] as i64),
        b'I' => Property::Integer(i32::from_le_bytes(reader.array()?) as i64),
        b'L' => Property::Integer(i64::from_le_bytes(reader.array()?)),
        b'F' => Property::Float(f32::from_le_bytes(reader.array()?) as f64),
        b'D' => Property::Float(f64::from_le_bytes(reader.array()?)),
        b'S' | b'R' => {
            let length = reader.u32()? as usize;
            let data = reader.take(length)?;
            if kind == b'S' {
                // "Name\0\x01Class": keep the name part.
                let name = data.split(|&b| b == 0).next().unwrap_or_default();
                Property::Text(String::from_utf8_lossy(name).into_owned())
            } else {
                Property::Other
            }
        }
        b'f' | b'd' | b'i' | b'l' | b'b' => {
            let count = reader.u32()? as usize;
            let encoding = reader.u32()?;
            let stored = reader.u32()? as usize;
            let width = match kind {
                b'd' | b'l' => 8,
                b'b' => 1,
                _ => 4,
            };
            let raw = reader.take(stored)?;
            let inflated;
            let data = match encoding {
                0 => raw,
                1 => {
                    inflated = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(raw, count * width)
                        .map_err(|_| IoError::malformed("corrupt compressed FBX array"))?;
                    &inflated
                }
                _ => return Err(IoError::malformed("unknown FBX array encoding")),
            };
            if data.len() < count * width {
                return Err(IoError::malformed("FBX array shorter than declared"));
            }
            let items = data.chunks_exact(width).take(count);
            match kind {
                b'f' => Property::Floats(items.map(|b| f32::from_le_bytes(b.try_into().expect("width")) as f64).collect()),
                b'd' => Property::Floats(items.map(|b| f64::from_le_bytes(b.try_into().expect("width"))).collect()),
                b'i' => Property::Integers(items.map(|b| i32::from_le_bytes(b.try_into().expect("width")) as i64).collect()),
                b'l' => Property::Integers(items.map(|b| i64::from_le_bytes(b.try_into().expect("width"))).collect()),
                _ => Property::Other,
            }
        }
        other => return Err(IoError::malformed(format!("unknown FBX property type {:?}", other as char))),
    })
}

// ---------------------------------------------------------------------------
// ASCII
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Token {
    /// `Name:`
    Key(String),
    Value(Property),
    Open,
    Close,
    Comma,
}

struct Lexer<'a> {
    text: &'a [u8],
    at: usize,
    line: usize,
    peeked: Option<Token>,
}

impl Lexer<'_> {
    fn error(&self, message: &str) -> IoError {
        IoError::Syntax { line: self.line, message: message.to_owned() }
    }

    fn peek(&mut self) -> Result<Option<&Token>, IoError> {
        if self.peeked.is_none() {
            self.peeked = self.read()?;
        }
        Ok(self.peeked.as_ref())
    }

    fn next(&mut self) -> Result<Option<Token>, IoError> {
        match self.peeked.take() {
            Some(token) => Ok(Some(token)),
            None => self.read(),
        }
    }

    fn read(&mut self) -> Result<Option<Token>, IoError> {
        loop {
            match self.text.get(self.at) {
                None => return Ok(None),
                Some(b'\n') => {
                    self.line += 1;
                    self.at += 1;
                }
                Some(b' ' | b'\t' | b'\r') => self.at += 1,
                Some(b';') => {
                    while self.text.get(self.at).is_some_and(|&c| c != b'\n') {
                        self.at += 1;
                    }
                }
                Some(_) => break,
            }
        }
        let start = self.at;
        let first = self.text[start];
        self.at += 1;
        Ok(Some(match first {
            b'{' => Token::Open,
            b'}' => Token::Close,
            b',' => Token::Comma,
            b'"' => {
                while self.text.get(self.at).is_some_and(|&c| c != b'"') {
                    self.line += usize::from(self.text[self.at] == b'\n');
                    self.at += 1;
                }
                if self.at >= self.text.len() {
                    return Err(self.error("unterminated string"));
                }
                self.at += 1;
                let text = String::from_utf8_lossy(&self.text[start + 1..self.at - 1]);
                // `Class::Name` in text files; keep the name part.
                Token::Value(Property::Text(text.rsplit("::").next().unwrap_or_default().to_owned()))
            }
            _ => {
                while self.text.get(self.at).is_some_and(|c| !b" \t\r\n,{}\";".contains(c)) {
                    self.at += 1;
                }
                let word = std::str::from_utf8(&self.text[start..self.at]).map_err(|_| self.error("not UTF-8"))?;
                if let Some(key) = word.strip_suffix(':') {
                    Token::Key(key.to_owned())
                } else if let Ok(integer) = word.parse::<i64>() {
                    Token::Value(Property::Integer(integer))
                } else if let Ok(float) = word.parse::<f64>() {
                    Token::Value(Property::Float(float))
                } else {
                    Token::Value(Property::Text(word.to_owned()))
                }
            }
        }))
    }
}

fn parse_ascii(bytes: &[u8]) -> Result<Node, IoError> {
    let mut lexer = Lexer { text: bytes, at: 0, line: 1, peeked: None };
    let mut root = Node::default();
    while let Some(token) = lexer.next()? {
        match token {
            Token::Key(name) => root.children.push(ascii_node(&mut lexer, name, 0)?),
            _ => return Err(lexer.error("expected a node name")),
        }
    }
    if root.children.is_empty() {
        return Err(IoError::malformed("not an FBX file"));
    }
    Ok(root)
}

/// Parses what follows `Name:`: a comma-separated property list and an
/// optional `{ … }` block of children.
fn ascii_node(lexer: &mut Lexer<'_>, name: String, depth: u32) -> Result<Node, IoError> {
    if depth > 64 {
        return Err(lexer.error("nodes nested too deeply"));
    }
    let mut node = Node { name, ..Node::default() };
    if matches!(lexer.peek()?, Some(Token::Value(_))) {
        loop {
            match lexer.next()? {
                Some(Token::Value(value)) => node.properties.push(value),
                _ => return Err(lexer.error("expected a value")),
            }
            if lexer.peek()? != Some(&Token::Comma) {
                break;
            }
            lexer.next()?;
        }
    }
    if lexer.peek()? == Some(&Token::Open) {
        lexer.next()?;
        loop {
            match lexer.next()? {
                Some(Token::Close) => break,
                Some(Token::Key(child)) => node.children.push(ascii_node(lexer, child, depth + 1)?),
                _ => return Err(lexer.error("expected a node or '}'")),
            }
        }
    }
    // Arrays are written `Name: *count { a: 1,2,3 }`: fold the values in.
    if matches!(node.properties.first(), Some(Property::Text(count)) if count.starts_with('*'))
        && node.children.len() == 1
        && node.children[0].name == "a"
    {
        let values = node.children.remove(0).numbers();
        node.properties = vec![Property::Floats(values)];
    }
    Ok(node)
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Local transform of a model: translation · pre-rotation · rotation ·
/// scale, with XYZ Euler angles in degrees.
fn model_transform(model: &Node) -> DMat4 {
    let vector = |name: &str, default: f64| {
        let values = model.property70(name).unwrap_or_default();
        match values.len() {
            n if n >= 3 => DVec3::new(values[n - 3], values[n - 2], values[n - 1]),
            _ => DVec3::splat(default),
        }
    };
    let rotation = |degrees: DVec3| {
        DMat4::from_rotation_z(degrees.z.to_radians())
            * DMat4::from_rotation_y(degrees.y.to_radians())
            * DMat4::from_rotation_x(degrees.x.to_radians())
    };
    DMat4::from_translation(vector("Lcl Translation", 0.0))
        * rotation(vector("PreRotation", 0.0))
        * rotation(vector("Lcl Rotation", 0.0))
        * DMat4::from_scale(vector("Lcl Scaling", 1.0))
}

/// Reads every mesh of an FBX file into one triangle mesh, in metres.
pub fn import_fbx(bytes: &[u8]) -> Result<ImportedMesh, IoError> {
    let root = parse_fbx(bytes)?;
    let objects = root.child("Objects").ok_or_else(|| IoError::malformed("FBX without Objects"))?;

    // child id → parent id
    let mut parent_of: HashMap<i64, i64> = HashMap::new();
    for connection in root.child("Connections").into_iter().flat_map(|c| c.children_named("C")) {
        if let (Some("OO"), Some(child), Some(parent)) = (connection.text(0), connection.integer(1), connection.integer(2)) {
            parent_of.entry(child).or_insert(parent);
        }
    }
    let models: HashMap<i64, DMat4> =
        objects.children_named("Model").filter_map(|m| Some((m.integer(0)?, model_transform(m)))).collect();
    let world_of = |geometry: i64| {
        let mut world = DMat4::IDENTITY;
        let mut current = parent_of.get(&geometry).copied();
        for _ in 0..256 {
            let Some(id) = current else { break };
            let Some(local) = models.get(&id) else { break };
            world = *local * world;
            current = parent_of.get(&id).copied();
        }
        world
    };

    let settings = root.child("GlobalSettings");
    let setting = |name: &str| settings.and_then(|s| s.property70(name)).and_then(|values| values.last().copied());
    // Centimetres per file unit.
    let to_metres = setting("UnitScaleFactor").filter(|s| *s > 0.0).unwrap_or(1.0) / 100.0;
    let z_up = setting("UpAxis") == Some(2.0);

    let mut mesh = ImportedMesh::default();
    for geometry in objects.children_named("Geometry") {
        let (Some(vertices), Some(polygons)) = (geometry.child("Vertices"), geometry.child("PolygonVertexIndex")) else {
            continue; // shapes, curves
        };
        let world = geometry.integer(0).map_or(DMat4::IDENTITY, world_of);
        let flip = world.determinant() < 0.0;
        let base = mesh.positions.len() as u32;
        let coordinates = vertices.numbers();
        mesh.positions.extend(coordinates.chunks_exact(3).map(|c| {
            let p = world.transform_point3(DVec3::new(c[0], c[1], c[2])) * to_metres;
            let p = if z_up { DVec3::new(p.x, p.z, -p.y) } else { p };
            Vec3::new(p.x as f32, p.y as f32, p.z as f32)
        }));
        let count = (mesh.positions.len() as u32 - base) as i64;
        let mut polygon: Vec<u32> = Vec::new();
        for index in polygons.numbers() {
            let index = index as i64;
            // The last corner of a polygon is stored bit-inverted.
            let (corner, last) = if index < 0 { (!index, true) } else { (index, false) };
            if corner >= count {
                return Err(IoError::malformed("FBX polygon index out of range"));
            }
            polygon.push(base + corner as u32);
            if last {
                for k in 1..polygon.len().saturating_sub(1) {
                    let t = [polygon[0], polygon[k], polygon[k + 1]];
                    mesh.triangles.push(if flip { [t[0], t[2], t[1]] } else { t });
                }
                polygon.clear();
            }
        }
    }
    Ok(mesh.finish())
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// How an exported FBX file is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FbxEncoding {
    /// What every FBX reader accepts, Blender included.
    #[default]
    Binary,
    /// Human-readable; read by Autodesk tools and game engines, but not
    /// by Blender.
    Ascii,
}

/// A property of a node to be written. Unlike [`Property`] this keeps the
/// exact storage type, which binary readers insist on.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    I32(i32),
    I64(i64),
    F64(f64),
    Text(String),
    /// An object name with its class, e.g. `Front` / `Geometry`.
    Name(String, &'static str),
    /// The bare letter some flags are written as in text files (`T`).
    Flag(char),
    I32s(Vec<i32>),
    F64s(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq)]
struct Out {
    name: &'static str,
    values: Vec<Value>,
    children: Vec<Out>,
}

fn node(name: &'static str, values: Vec<Value>, children: Vec<Out>) -> Out {
    Out { name, values, children }
}

fn text(value: &str) -> Value {
    Value::Text(value.to_owned())
}

/// `P: "name", "type", "label", "flags", values…` of a `Properties70` list.
fn property70(name: &str, kind: &str, label: &str, flags: &str, values: Vec<Value>) -> Out {
    let mut all = vec![text(name), text(kind), text(label), text(flags)];
    all.extend(values);
    node("P", all, Vec::new())
}

fn write_ascii(out: &mut String, item: &Out, depth: usize) {
    let indent = "\t".repeat(depth);
    let _ = write!(out, "{indent}{}: ", item.name);
    let mut array: Option<String> = None;
    for (i, value) in item.values.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match value {
            Value::I32(v) => _ = write!(out, "{v}"),
            Value::I64(v) => _ = write!(out, "{v}"),
            Value::F64(v) => _ = write!(out, "{v}"),
            Value::Text(v) => _ = write!(out, "\"{v}\""),
            Value::Name(name, class) => _ = write!(out, "\"{class}::{name}\""),
            Value::Flag(c) => out.push(*c),
            Value::I32s(list) => {
                let _ = write!(out, "*{}", list.len());
                array = Some(list.iter().map(i32::to_string).collect::<Vec<_>>().join(","));
            }
            Value::F64s(list) => {
                let _ = write!(out, "*{}", list.len());
                array = Some(list.iter().map(f64::to_string).collect::<Vec<_>>().join(","));
            }
        }
    }
    if array.is_none() && item.children.is_empty() {
        out.push('\n');
        return;
    }
    out.push_str(" {\n");
    if let Some(values) = array {
        let _ = writeln!(out, "{indent}\ta: {values}");
    }
    for child in &item.children {
        write_ascii(out, child, depth + 1);
    }
    let _ = writeln!(out, "{indent}}}");
}

/// Length of the all-zero record that ends a list of nodes (FBX < 7.5).
const NULL_RECORD: usize = 13;

fn write_binary(out: &mut Vec<u8>, item: &Out) {
    let start = out.len();
    out.extend([0u8; 12]); // end offset, property count, property bytes
    out.push(item.name.len() as u8);
    out.extend(item.name.as_bytes());
    let properties_start = out.len();
    for value in &item.values {
        match value {
            Value::I32(v) => {
                out.push(b'I');
                out.extend(v.to_le_bytes());
            }
            Value::I64(v) => {
                out.push(b'L');
                out.extend(v.to_le_bytes());
            }
            Value::F64(v) => {
                out.push(b'D');
                out.extend(v.to_le_bytes());
            }
            Value::Text(_) | Value::Name(..) | Value::Flag(_) => {
                let bytes = match value {
                    Value::Text(v) => v.clone().into_bytes(),
                    // Name and class are separated by 0x00 0x01.
                    Value::Name(name, class) => [name.as_bytes(), &[0, 1], class.as_bytes()].concat(),
                    Value::Flag(c) => vec![*c as u8],
                    _ => unreachable!(),
                };
                out.push(b'S');
                out.extend((bytes.len() as u32).to_le_bytes());
                out.extend(bytes);
            }
            Value::I32s(list) => {
                out.push(b'i');
                // count, encoding (0 = uncompressed), byte length
                out.extend((list.len() as u32).to_le_bytes());
                out.extend(0u32.to_le_bytes());
                out.extend((list.len() as u32 * 4).to_le_bytes());
                out.extend(list.iter().flat_map(|v| v.to_le_bytes()));
            }
            Value::F64s(list) => {
                out.push(b'd');
                out.extend((list.len() as u32).to_le_bytes());
                out.extend(0u32.to_le_bytes());
                out.extend((list.len() as u32 * 8).to_le_bytes());
                out.extend(list.iter().flat_map(|v| v.to_le_bytes()));
            }
        }
    }
    let property_bytes = out.len() - properties_start;
    if !item.children.is_empty() {
        for child in &item.children {
            write_binary(out, child);
        }
        out.extend([0u8; NULL_RECORD]);
    }
    let end = out.len() as u32;
    out[start..start + 4].copy_from_slice(&end.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(item.values.len() as u32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&(property_bytes as u32).to_le_bytes());
}

/// The node tree of a garment: one mesh per pattern piece with normals, a
/// UV set and a material per fabric. Centimetres (the FBX default), Y up.
fn garment_tree(garment: &GarmentExport<'_>) -> Vec<Out> {
    const TO_CM: f32 = 100.0;
    let mesh = garment.mesh;
    let uvs = garment.uvs();
    let groups = garment.groups();
    let mut materials: Vec<String> = Vec::new();
    for (_, fabric, _) in &groups {
        if !materials.contains(fabric) {
            materials.push(fabric.clone());
        }
    }
    let int = |name: &str, value: i32| property70(name, "int", "Integer", "", vec![Value::I32(value)]);
    let vector = |name: &str, v: [f64; 3]| property70(name, name, "", "A", v.map(Value::F64).to_vec());

    let header = node(
        "FBXHeaderExtension",
        vec![],
        vec![
            node("FBXHeaderVersion", vec![Value::I32(1003)], vec![]),
            node("FBXVersion", vec![Value::I32(7400)], vec![]),
            node("Creator", vec![text("clo360")], vec![]),
        ],
    );
    let settings = node(
        "GlobalSettings",
        vec![],
        vec![
            node("Version", vec![Value::I32(1000)], vec![]),
            node(
                "Properties70",
                vec![],
                vec![
                    int("UpAxis", 1),
                    int("UpAxisSign", 1),
                    int("FrontAxis", 2),
                    int("FrontAxisSign", 1),
                    int("CoordAxis", 0),
                    int("CoordAxisSign", 1),
                    // Centimetres per unit.
                    property70("UnitScaleFactor", "double", "Number", "", vec![Value::F64(1.0)]),
                    property70("OriginalUnitScaleFactor", "double", "Number", "", vec![Value::F64(1.0)]),
                ],
            ),
        ],
    );
    let object_type = |kind: &str, count: usize| node("ObjectType", vec![text(kind)], vec![node("Count", vec![Value::I32(count as i32)], vec![])]);
    let definitions = node(
        "Definitions",
        vec![],
        vec![
            node("Version", vec![Value::I32(100)], vec![]),
            node("Count", vec![Value::I32((2 * groups.len() + materials.len()) as i32)], vec![]),
            object_type("Geometry", groups.len()),
            object_type("Model", groups.len()),
            object_type("Material", materials.len()),
        ],
    );

    let (geometry_id, model_id, material_id) = (|i: usize| 1000 + i as i64, |i: usize| 2000 + i as i64, |i: usize| 3000 + i as i64);
    let mut objects = Vec::new();
    let mut connections = Vec::new();
    let connect = |child: i64, parent: i64| node("C", vec![text("OO"), Value::I64(child), Value::I64(parent)], vec![]);
    for (index, (piece, fabric, triangles)) in groups.iter().enumerate() {
        // Each piece gets its own vertex numbering.
        let mut local: HashMap<u32, i32> = HashMap::new();
        let mut order: Vec<u32> = Vec::new();
        for &v in triangles.iter() {
            local.entry(v).or_insert_with(|| {
                order.push(v);
                order.len() as i32 - 1
            });
        }
        let doubles = |values: &mut dyn Iterator<Item = f32>| Value::F64s(values.map(f64::from).collect());
        let layer_element = |kind: &'static str| {
            node("LayerElement", vec![], vec![node("Type", vec![text(kind)], vec![]), node("TypedIndex", vec![Value::I32(0)], vec![])])
        };
        let mapping = |name: &str, mapping: &str, reference: &str| {
            vec![
                node("Version", vec![Value::I32(101)], vec![]),
                node("Name", vec![text(name)], vec![]),
                node("MappingInformationType", vec![text(mapping)], vec![]),
                node("ReferenceInformationType", vec![text(reference)], vec![]),
            ]
        };
        let with = |mut head: Vec<Out>, tail: Vec<Out>| {
            head.extend(tail);
            head
        };
        objects.push(node(
            "Geometry",
            vec![Value::I64(geometry_id(index)), Value::Name(piece.clone(), "Geometry"), text("Mesh")],
            vec![
                node("Vertices", vec![doubles(&mut order.iter().flat_map(|&v| (mesh.positions[v as usize] * TO_CM).to_array()))], vec![]),
                // The last corner of each polygon is stored bit-inverted.
                node(
                    "PolygonVertexIndex",
                    vec![Value::I32s(triangles.chunks_exact(3).flat_map(|t| [local[&t[0]], local[&t[1]], !local[&t[2]]]).collect())],
                    vec![],
                ),
                node("GeometryVersion", vec![Value::I32(124)], vec![]),
                node(
                    "LayerElementNormal",
                    vec![Value::I32(0)],
                    with(
                        mapping("", "ByVertice", "Direct"),
                        vec![node("Normals", vec![doubles(&mut order.iter().flat_map(|&v| mesh.normals[v as usize].to_array()))], vec![])],
                    ),
                ),
                node(
                    "LayerElementUV",
                    vec![Value::I32(0)],
                    with(
                        mapping("UVMap", "ByPolygonVertex", "IndexToDirect"),
                        vec![
                            node("UV", vec![doubles(&mut order.iter().flat_map(|&v| uvs[v as usize].to_array()))], vec![]),
                            node("UVIndex", vec![Value::I32s(triangles.iter().map(|v| local[v]).collect())], vec![]),
                        ],
                    ),
                ),
                node(
                    "LayerElementMaterial",
                    vec![Value::I32(0)],
                    with(mapping("", "AllSame", "IndexToDirect"), vec![node("Materials", vec![Value::I32s(vec![0])], vec![])]),
                ),
                node(
                    "Layer",
                    vec![Value::I32(0)],
                    vec![
                        node("Version", vec![Value::I32(100)], vec![]),
                        layer_element("LayerElementNormal"),
                        layer_element("LayerElementUV"),
                        layer_element("LayerElementMaterial"),
                    ],
                ),
            ],
        ));
        objects.push(node(
            "Model",
            vec![Value::I64(model_id(index)), Value::Name(piece.clone(), "Model"), text("Mesh")],
            vec![
                node("Version", vec![Value::I32(232)], vec![]),
                node(
                    "Properties70",
                    vec![],
                    vec![vector("Lcl Translation", [0.0; 3]), vector("Lcl Rotation", [0.0; 3]), vector("Lcl Scaling", [1.0; 3])],
                ),
                node("Shading", vec![Value::Flag('T')], vec![]),
                // Cloth is seen from both sides.
                node("Culling", vec![text("CullingOff")], vec![]),
            ],
        ));
        let material = materials.iter().position(|m| m == fabric).unwrap_or(0);
        connections.extend([
            connect(model_id(index), 0),
            connect(geometry_id(index), model_id(index)),
            connect(material_id(material), model_id(index)),
        ]);
    }
    for (index, name) in materials.iter().enumerate() {
        let safe = |fabric: &str| fabric.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect::<String>();
        let color = garment.document.fabrics().iter().find(|f| safe(&f.name) == *name).map_or([0.6; 3], |f| f.base_color);
        objects.push(node(
            "Material",
            vec![Value::I64(material_id(index)), Value::Name(name.clone(), "Material"), text("")],
            vec![
                node("Version", vec![Value::I32(102)], vec![]),
                node("ShadingModel", vec![text("lambert")], vec![]),
                node(
                    "Properties70",
                    vec![],
                    vec![property70("DiffuseColor", "Color", "", "A", color.map(|c| Value::F64(c as f64)).to_vec())],
                ),
            ],
        ));
    }
    vec![header, settings, definitions, node("Objects", vec![], objects), node("Connections", vec![], connections)]
}

/// Writes the draped garment as FBX 7.4: one mesh per pattern piece with
/// normals, a UV set and a material per fabric. Units are centimetres (the
/// FBX default), Y up.
pub fn export_garment_fbx(garment: &GarmentExport<'_>, encoding: FbxEncoding) -> Vec<u8> {
    let tree = garment_tree(garment);
    match encoding {
        FbxEncoding::Ascii => {
            let mut out = String::from("; FBX 7.4.0 project file\n; Created by clo360\n");
            for item in &tree {
                write_ascii(&mut out, item, 0);
            }
            out.into_bytes()
        }
        FbxEncoding::Binary => {
            let mut out = BINARY_MAGIC.to_vec();
            out.extend([0x1A, 0x00]);
            out.extend(7400u32.to_le_bytes());
            for item in &tree {
                write_binary(&mut out, item);
            }
            out.extend([0u8; NULL_RECORD]);
            // Footer: an id, padding to a 16-byte boundary, the version
            // again, 120 zero bytes and a closing magic number.
            out.extend([0xFA, 0xBC, 0xAB, 0x09, 0xD0, 0xC8, 0xD4, 0x66, 0xB1, 0x76, 0xFB, 0x83, 0x1C, 0xF7, 0x26, 0x7E]);
            out.extend([0u8; 4]);
            let padding = (16 - out.len() % 16) % 16;
            out.extend(std::iter::repeat_n(0u8, if padding == 0 { 16 } else { padding }));
            out.extend(7400u32.to_le_bytes());
            out.extend([0u8; 120]);
            out.extend([0xF8, 0x5A, 0x8C, 0x6A, 0xDE, 0xF5, 0xD9, 0x7E, 0xEC, 0xE9, 0x0C, 0xE3, 0x75, 0x8F, 0x29, 0x0B]);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASCII: &str = r#"; FBX 7.4.0 project file
GlobalSettings:  {
    Properties70:  {
        P: "UpAxis", "int", "Integer", "",1
        P: "UnitScaleFactor", "double", "Number", "",1
    }
}
Objects:  {
    Geometry: 10, "Geometry::Quad", "Mesh" {
        Vertices: *12 {
            a: 0,0,0,100,0,0,
100,100,0,0,100,0
        }
        PolygonVertexIndex: *4 {
            a: 0,1,2,-4
        }
    }
    Model: 20, "Model::Quad", "Mesh" {
        Properties70:  {
            P: "Lcl Translation", "Lcl Translation", "", "A",0,50,0
            P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,90
        }
    }
}
Connections:  {
    C: "OO",20,0
    C: "OO",10,20
}
"#;

    #[test]
    fn ascii_files_are_read_with_units_and_model_transforms() {
        let mesh = import_fbx(ASCII.as_bytes()).unwrap();
        // A quad becomes two triangles.
        assert_eq!(mesh.triangles, vec![[0, 1, 2], [0, 2, 3]]);
        // Centimetres → metres; rotated 90° about z, then lifted 0.5 m.
        let expected = [Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.0, 1.5, 0.0), Vec3::new(-1.0, 1.5, 0.0), Vec3::new(-1.0, 0.5, 0.0)];
        for (found, expected) in mesh.positions.iter().zip(expected) {
            assert!((*found - expected).length() < 1e-5, "{found} vs {expected}");
        }
        assert!((mesh.normals[0] - Vec3::Z).length() < 1e-5);
    }

    // --- a minimal binary writer, to test the binary reader -----------------

    enum P<'a> {
        Long(i64),
        Text(&'a str),
        Doubles(&'a [f64], bool),
        Ints(&'a [i32]),
    }

    fn node(name: &str, properties: &[P<'_>], children: &[Vec<u8>], offset: usize) -> Vec<u8> {
        let mut props = Vec::new();
        for property in properties {
            match property {
                P::Long(value) => {
                    props.push(b'L');
                    props.extend(value.to_le_bytes());
                }
                P::Text(text) => {
                    props.push(b'S');
                    props.extend((text.len() as u32).to_le_bytes());
                    props.extend(text.as_bytes());
                }
                P::Doubles(values, compressed) => {
                    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let stored = if *compressed { miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6) } else { raw };
                    props.push(b'd');
                    props.extend((values.len() as u32).to_le_bytes());
                    props.extend(u32::from(*compressed).to_le_bytes());
                    props.extend((stored.len() as u32).to_le_bytes());
                    props.extend(stored);
                }
                P::Ints(values) => {
                    props.push(b'i');
                    props.extend((values.len() as u32).to_le_bytes());
                    props.extend(0u32.to_le_bytes());
                    props.extend((values.len() as u32 * 4).to_le_bytes());
                    props.extend(values.iter().flat_map(|v| v.to_le_bytes()));
                }
            }
        }
        let children_bytes: usize = children.iter().map(Vec::len).sum::<usize>() + if children.is_empty() { 0 } else { 13 };
        let length = 13 + name.len() + props.len() + children_bytes;
        let mut out = Vec::new();
        out.extend(((offset + length) as u32).to_le_bytes());
        out.extend((properties.len() as u32).to_le_bytes());
        out.extend((props.len() as u32).to_le_bytes());
        out.push(name.len() as u8);
        out.extend(name.as_bytes());
        out.extend(props);
        for child in children {
            out.extend(child);
        }
        if !children.is_empty() {
            out.extend([0u8; 13]);
        }
        out
    }

    #[test]
    fn binary_files_with_compressed_arrays_are_read() {
        // Offsets are absolute, so nodes are laid out innermost first with
        // their final position known.
        let header = BINARY_MAGIC.len() + 2 + 4;
        let vertices = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let geometry_props = [P::Long(7), P::Text("Tri\0\x01Geometry"), P::Text("Mesh")];
        // Sizes are needed before offsets: build once to measure.
        let build = |objects_at: usize| {
            let geometry_at = objects_at + 13 + "Objects".len();
            let geometry_head = 13 + "Geometry".len() + node("", &geometry_props, &[], 0).len() - 13;
            let vertices_at = geometry_at + geometry_head;
            let vertices_node = node("Vertices", &[P::Doubles(&vertices, true)], &[], vertices_at);
            let polygons_node = node("PolygonVertexIndex", &[P::Ints(&[0, 1, -3])], &[], vertices_at + vertices_node.len());
            let geometry = node("Geometry", &geometry_props, &[vertices_node, polygons_node], geometry_at);
            node("Objects", &[], &[geometry], objects_at)
        };
        let mut file = BINARY_MAGIC.to_vec();
        file.extend([0x1A, 0x00]);
        file.extend(7400u32.to_le_bytes());
        assert_eq!(file.len(), header);
        file.extend(build(header));
        file.extend([0u8; 13]);

        let root = parse_fbx(&file).unwrap();
        let geometry = root.child("Objects").unwrap().child("Geometry").unwrap();
        assert_eq!(geometry.text(1), Some("Tri"));
        let mesh = import_fbx(&file).unwrap();
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
        // No GlobalSettings: centimetres.
        assert!((mesh.positions[1] - Vec3::new(0.02, 0.0, 0.0)).length() < 1e-7);
    }
}
