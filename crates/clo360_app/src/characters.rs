//! The character library: avatars saved under a name, with their build,
//! tape measurements and looks, kept for every project.
//!
//! Each character is one JSON file in `$XDG_DATA_HOME/clo360/characters`
//! (`~/.local/share/clo360/characters`), or in `CLO360_CHARACTERS` if that
//! is set. The Avatar window lists them and puts one on with a click.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::project::AvatarFile;

const FORMAT: &str = "clo360-character";
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Character {
    format: String,
    version: u32,
    pub name: String,
    /// Measured height in metres when saved (shown in the list).
    #[serde(default)]
    pub height: f32,
    pub avatar: AvatarFile,
}

impl Character {
    pub fn new(name: &str, height: f32, avatar: AvatarFile) -> Self {
        Self {
            format: FORMAT.to_owned(),
            version: VERSION,
            name: name.trim().to_owned(),
            height,
            avatar,
        }
    }

    /// What the Avatar window shows and puts on.
    pub fn entry(&self) -> app_ui::CharacterEntry {
        app_ui::CharacterEntry {
            name: self.name.clone(),
            height: self.height,
            params: self.avatar.params(),
            targets: self.avatar.targets(),
            appearance: self.avatar.appearance(),
        }
    }
}

/// Where the library lives.
pub fn folder() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("CLO360_CHARACTERS") {
        return Some(PathBuf::from(dir));
    }
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(data.join("clo360").join("characters"))
}

/// A file name for a character's name: letters and digits kept, the rest
/// turned into `_`. Names differing only in case share one file.
fn file_name(name: &str) -> String {
    let stem: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                '_'
            }
        })
        .collect();
    format!(
        "{}.json",
        if stem.is_empty() {
            "character".into()
        } else {
            stem
        }
    )
}

/// Every character in `folder`, sorted by name. Unreadable files are skipped.
pub fn list(folder: &Path) -> Vec<Character> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut characters: Vec<Character> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|path| {
            let text = std::fs::read_to_string(&path).ok()?;
            let character: Character = serde_json::from_str(&text)
                .map_err(|e| log::warn!("skipping character {}: {e}", path.display()))
                .ok()?;
            (character.format == FORMAT && character.version <= VERSION).then_some(character)
        })
        .collect();
    characters.sort_by_key(|c| c.name.to_lowercase());
    characters
}

/// Saves (or replaces) the character with this name.
pub fn save(folder: &Path, character: &Character) -> anyhow::Result<()> {
    anyhow::ensure!(!character.name.is_empty(), "a character needs a name");
    std::fs::create_dir_all(folder)
        .with_context(|| format!("cannot create {}", folder.display()))?;
    let path = folder.join(file_name(&character.name));
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_string_pretty(character)?)
        .with_context(|| format!("cannot write {}", temporary.display()))?;
    std::fs::rename(&temporary, &path)
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Removes the character with this name.
pub fn delete(folder: &Path, name: &str) -> anyhow::Result<()> {
    let path = folder.join(file_name(name));
    std::fs::remove_file(&path).with_context(|| format!("cannot delete {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use avatar::{Appearance, BodyParams, HairStyle, Measure, PoseParams};

    #[test]
    fn characters_are_saved_listed_replaced_and_deleted() {
        let folder = std::env::temp_dir().join(format!("clo360_characters_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        assert!(list(&folder).is_empty());

        let mut params = BodyParams::male();
        params.weight = 0.8;
        let mut targets = [None; Measure::ALL.len()];
        targets[Measure::Waist as usize] = Some(0.92);
        let mut looks = Appearance::default();
        looks.hair.style = HairStyle::Ponytail;
        let avatar = AvatarFile::new(&params, &PoseParams::default(), &targets, &looks);
        save(
            &folder,
            &Character::new("  Tom / tall ", 1.87, avatar.clone()),
        )
        .unwrap();
        save(
            &folder,
            &Character::new(
                "Anna",
                1.64,
                AvatarFile::new(
                    &BodyParams::female(),
                    &PoseParams::default(),
                    &[None; Measure::ALL.len()],
                    &Appearance::default(),
                ),
            ),
        )
        .unwrap();

        let all = list(&folder);
        assert_eq!(
            all.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["Anna", "Tom / tall"]
        );
        let tom = all[1].entry();
        assert_eq!(tom.params, params);
        assert_eq!(tom.targets, targets);
        assert_eq!(tom.appearance, looks);

        // Same name: replaced, not added.
        save(&folder, &Character::new("Tom / tall", 1.90, avatar)).unwrap();
        assert_eq!(list(&folder).len(), 2);
        assert_eq!(list(&folder)[1].height, 1.90);

        delete(&folder, "Anna").unwrap();
        assert_eq!(list(&folder).len(), 1);
        std::fs::remove_dir_all(&folder).unwrap();
    }
}
