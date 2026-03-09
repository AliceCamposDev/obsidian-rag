use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use stopwords::{Language, Spark, Stopwords};
use urlencoding::decode;
use uuid::Uuid;

#[derive(Debug)]
pub struct NoteMap {
    pub notes: HashMap<Uuid, Note>,
    pub nodes: usize,
    pub version: String,
}

impl NoteMap {
    pub fn new() -> Self {
        Self {
            notes: HashMap::new(),
            nodes: 0,
            version: String::from("1.0"),
        }
    }
    pub fn add_note(&mut self, note: Note) -> bool {
        self.notes.insert(note.note_id, note);
        self.nodes = self.nodes + 1;
        true
    }
    #[allow(dead_code)]
    pub fn get_version(&self) -> String {
        self.version.clone()
    }
    //TODO
    // pub fn reload_bounds(&mut self) {

    // }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawNote {
    pub raw_id: Uuid,
    pub raw_content: String,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub note_id: Uuid,
    pub raw_note: RawNote,
    pub chunks: HashMap<Uuid, ContentChunk>,
    pub rel_path: PathBuf,
    pub title: String,
    pub wiki_links: Vec<Uuid>,
    pub back_links: Vec<Uuid>,
    pub tags: Vec<String>,
    pub str_wiki_links: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentChunk {
    pub chunk_id: Uuid,
    pub keywords: Vec<String>,
    pub content: String,
    pub size: usize,
}

impl Note {
    pub fn new(raw_note: RawNote, rel_path: PathBuf) -> Self {
        let title = Self::get_note_title(&rel_path);
        let str_wiki_links = Self::get_wiki_links(&raw_note);
        let chunks = Self::gen_chunks(&raw_note);

        Self {
            tags: vec![String::from("none")],
            note_id: Uuid::new_v4(),
            raw_note,
            chunks: chunks,
            rel_path: rel_path,
            title,
            wiki_links: Vec::new(),
            back_links: Vec::new(),
            str_wiki_links: str_wiki_links,
        }
    }
    fn get_note_title(path: &PathBuf) -> String {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("no_title")
            .to_string()
    }
    fn get_wiki_links(raw: &RawNote) -> Vec<String> {
        let content = &raw.clone().raw_content;
        let mut links = Vec::new();

        let wiki_re = Regex::new(r"\[\[([^\]]+)\]\]").unwrap();
        for cap in wiki_re.captures_iter(content) {
            let target = &cap[1];
            let target = target
                .split(|c| c == '#' || c == '|')
                .next()
                .unwrap_or(target);
            let target = target.trim_end_matches(".md");
            let target = decode(target).unwrap_or_else(|_| target.into()).to_string();
            links.push(target);
        }

        let md_re = Regex::new(r"\[[^\]]+\]\(([^)]+)\)").unwrap();
        for cap in md_re.captures_iter(content) {
            let target = &cap[1];
            if target.contains("://") {
                continue;
            }
            let target = target.trim_end_matches(".md");
            let target = decode(target).unwrap_or_else(|_| target.into()).to_string();
            links.push(target);
        }
        links
    }
    fn gen_chunks(raw: &RawNote) -> HashMap<Uuid, ContentChunk> {
        let content = raw.clone().raw_content;
        let content_len = content.len();
        let mut result = HashMap::new();
        let chunk_id = Uuid::new_v4();
        let stops: HashSet<String> = Spark::stopwords(Language::English)
            .unwrap()
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        let mut tokens: Vec<String> = content
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();

        tokens.retain(|w| !stops.contains(w));
        let keywords: Vec<String> = tokens;

        // println!("{:?}", keywords);
        let chunk = ContentChunk {
            chunk_id: chunk_id,
            keywords: keywords,
            content: content,
            size: content_len,
        };
        result.insert(chunk.chunk_id, chunk);
        result
    }
}

pub fn parse_note(path: &PathBuf) -> Note {
    let content = fs::read_to_string(&path).expect("err reading file");
    let now = Utc::now();

    let raw_note = RawNote {
        raw_id: Uuid::new_v4(),
        raw_content: content.clone(),
        created_at: now,
        modified_at: now,
        size: content.len(),
    };

    let chunk = ContentChunk {
        chunk_id: Uuid::new_v4(),
        keywords: Vec::new(),
        content: content.clone(),
        size: content.len(),
    };

    let mut chunks: HashMap<Uuid, ContentChunk> = HashMap::new();
    chunks.insert(chunk.chunk_id, chunk);

    let rel_path = path;

    let note = Note::new(raw_note, rel_path.clone());
    // println!("title: {}", note.title);
    // println!("path: {}", note.rel_path.to_string_lossy().to_string());
    // println!("size: {}", note.raw_note.size.to_string());
    note
}
