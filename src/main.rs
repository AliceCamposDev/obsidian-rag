mod load_vault;
mod models;

use load_vault::load_vault::load_vault_fn;
use models::note_structure::NoteMap;

fn main() {
    // load_vault_fn(&std::path::PathBuf::from(r"C:\workspace\rpg-guerrilha-urbana-back\fastapi\book"));
    let map: NoteMap = load_vault_fn(&std::path::PathBuf::from(
        r"C:\workspace\nlp_search\Bible Study Kit (v1)",
    ));

    for val in map.notes.values() {
        println!("{:?}", val.note_id);
    }

    
}
