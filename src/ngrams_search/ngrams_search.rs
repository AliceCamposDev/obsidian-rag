use crate::models::note_structure::NoteMap;
use std::collections::HashMap;
use uuid::Uuid;
use crate::utils::text_preprocess::{tokenize};

pub fn nlp_search_fn(query: &str, note_map: &NoteMap) {
    println!("Query: {}", query);
    println!("Searching in {} notes...", note_map.notes.len());

    fn extract_np_candidates(text: &str) -> Vec<String> {
        let tokens = tokenize(text);
        let mut phrases = Vec::new();

        for i in 0..tokens.len() {
            for j in i + 2..=(i + 4).min(tokens.len()) {
                let phrase = tokens[i..j].join(" ");
                phrases.push(phrase);
            }
        }

        phrases
    }
    fn expand_np(np: &str) -> Vec<String> {
        let words: Vec<&str> = np.split_whitespace().collect();
        let mut phrases = Vec::new();

        if words.len() < 2 {
            return phrases;
        }

        for i in 0..words.len() {
            for j in i + 1..=words.len() {
                phrases.push(words[i..j].join(" "));
            }
        }

        phrases
    }
    let mut index: HashMap<String, HashMap<Uuid, usize>> = HashMap::new();
    let note_map_notes_cl = note_map.notes.clone();
    let mut cont = 0;
    for (id, note) in note_map_notes_cl {
       let text: String = note
        .chunks
        .values()
        .flat_map(|chunk| chunk.keywords.clone())
        .collect::<Vec<String>>()
        .join(" ");

        let doc_id = id;
        let candidates = extract_np_candidates(&text);

        for np in candidates {
            let phrases = expand_np(&np);

            for phrase in phrases {
                let entry: &mut HashMap<Uuid, usize> =
                    index.entry(phrase).or_insert_with(HashMap::new);

                *entry.entry(doc_id).or_insert(0) += 1;
            }
        }
        cont += 1;
        print!("note {}/{}\r", cont, note_map.notes.len());
    }
    // print!("{:?}", index);
    let mut scores: HashMap<Uuid, usize> = HashMap::new();
    let candidates = extract_np_candidates(query);
    for np in candidates {
        let phrases = expand_np(&np);

        for phrase in phrases {
            if let Some(docs) = index.get(&phrase) {
                for (doc, weight) in docs {
                    *scores.entry(*doc).or_insert(0) += *weight;
                }
            }
        }
    }

    let mut results: Vec<(Uuid, usize)> = scores.into_iter().collect();

    results.sort_by(|a, b| b.1.cmp(&a.1));

    for (i, (doc, score)) in results.iter().enumerate() {
        if i >= 1 {
            break;
        }
        println!("Document {} | score {}", doc, score);
        println!(
            "Content: {}",
            note_map.notes.get(doc).unwrap().raw_note.raw_content
        );
    }
}
