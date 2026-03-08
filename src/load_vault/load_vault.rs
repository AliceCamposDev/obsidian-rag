use rayon::prelude::*;
use std::fs;
use std::sync::Mutex;
use walkdir::WalkDir;
use std::path::PathBuf;
use crate::models::note_structure::parse_note;

use crate::models::note_structure::NoteMap;

pub fn load_vault_fn(path: &PathBuf) ->  NoteMap {
    let entries: Vec<std::path::PathBuf> = WalkDir::new(path)
        .into_iter()
        .filter_map(|entry_res: Result<walkdir::DirEntry, walkdir::Error>| {
            let entry: walkdir::DirEntry = entry_res.ok()?;
            let ft = entry.file_type();
            if ft.is_file() {
                if let Some(ext) = entry.path().extension() {
                    if ext.eq_ignore_ascii_case("md") {
                        return Some(entry.path().to_path_buf());
                    }
                }
                None
            } else {
                None
            }
        })
        .collect();

    // println!("{:#?}", entries);

    let map = Mutex::new(NoteMap::new());

    entries.par_iter().for_each(|path| {
        // println!("Thread: {:?}", thread::current().id());
        if path.is_file() {
            match fs::read_to_string(path) {
                Ok(_) => {
                    let note = parse_note(path);
                    map.lock().unwrap().add_note(note);
                }
                Err(_) => {
                    eprintln!("Err, skipping {}", path.display());
                }
            }
        }
    });

    map.into_inner().unwrap()
    // for note in map.notes.values() {
    //     println!("{}", note.title);
    // }
    // println!("{}", map.nodes.to_string());
    // print!("{}", map.get_version());


}


