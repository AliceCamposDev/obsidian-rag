mod load_vault;
mod models;
mod ngrams_search;

use ngrams_search::ngrams_search::nlp_search_fn;
use load_vault::load_vault::load_vault_fn;
use models::note_structure::NoteMap;

fn main() {
    // load_vault_fn(&std::path::PathBuf::from(r"C:\workspace\rpg-guerrilha-urbana-back\fastapi\book"));
    let map: NoteMap = load_vault_fn(&std::path::PathBuf::from(
        r"C:\workspace\nlp_search\Bible Study Kit (v1)",
    ));



    let query = "Why do you kick at my sacrifice and at my offering";
    
    nlp_search_fn(&query, &map);

}
