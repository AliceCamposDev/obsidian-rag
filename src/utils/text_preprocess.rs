use stopwords::{Language, Spark, Stopwords};
use regex::Regex;
use std::collections::HashSet;
use urlencoding::decode;


pub fn normalize(text: &str) -> String {
    text.to_lowercase()
        .replace(",", "")
        .replace(".", "")
        .replace(";", "")
        .replace("!", "")
        .replace("?", "")
}

pub fn tokenize(text: &str) -> Vec<String> {
    normalize(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

pub fn remove_stopwords(tokens: Vec<String>) -> Vec<String> {
    let stops: HashSet<String> =  Spark::stopwords(Language::English)
        .unwrap()
        .iter()
        .map(|s| s.to_lowercase())
        .collect();

    tokens.into_iter().filter(|w| !stops.contains(&w.to_lowercase())).collect()
}

pub fn get_wiki_links(content: &str) -> Vec<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn normalize_lowercase() {
        assert_eq!(normalize("Hello"), "hello");
    }

    #[test]
    fn normalize_removes_punctuation() {
        assert_eq!(normalize("Hello, world."), "hello world");
        assert_eq!(normalize("a;b,c."), "abc");
        assert_eq!(normalize("Hello,; ??world.!"), "hello world");
    }

    #[test]
    fn tokenize_simple() {
        assert_eq!(tokenize("Hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_with_punctuation() {
        assert_eq!(tokenize("Hello, world!"), vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_extra_whitespace() {
        assert_eq!(tokenize("Hello   world"), vec!["hello", "world"]);
        assert_eq!(tokenize("  Hello world  "), vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_combined_punctuation() {
        assert_eq!(tokenize("Hello,; world."), vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_non_ascii() {
        assert_eq!(tokenize("café"), vec!["café"]);
        assert_eq!(tokenize("Æsop"), vec!["æsop"]);
    }

    #[test]
    fn remove_stopwords_filters_common() {
        let tokens = vec![
            "the".to_string(),
            "cat".to_string(),
            "sat".to_string(),
            "on".to_string(),
            "the".to_string(),
            "mat".to_string(),
        ];
        let result = remove_stopwords(tokens);
        assert_eq!(result, vec!["cat", "sat", "mat"]);
    }

    #[test]
    fn remove_stopwords_no_stopwords() {
        let tokens = vec!["cat".to_string(), "sat".to_string()];
        let result = remove_stopwords(tokens);
        assert_eq!(result, vec!["cat", "sat"]);
    }

    #[test]
    fn remove_stopwords_empty() {
        let tokens: Vec<String> = vec![];
        let result = remove_stopwords(tokens);
        assert!(result.is_empty());
    }

    #[test]
    fn remove_stopwords_case_insensitive() {
        let tokens = vec!["The".to_string(), "Cat".to_string()];
        let result = remove_stopwords(tokens);
        assert_eq!(result, vec!["Cat"]);
    }


    #[test]
    fn wiki_links_simple() {
        let content = "[[Page]]";
        assert_eq!(get_wiki_links(content), vec!["Page"]);
    }

    #[test]
    fn wiki_links_with_section() {
        let content = "[[Page#section]]";
        assert_eq!(get_wiki_links(content), vec!["Page"]);
    }

    #[test]
    fn wiki_links_with_pipe() {
        let content = "[[Page|display]]";
        assert_eq!(get_wiki_links(content), vec!["Page"]);
    }

    #[test]
    fn wiki_links_with_pipe_and_section() {
        let content = "[[Page#section|display]]";
        assert_eq!(get_wiki_links(content), vec!["Page"]);
    }

    #[test]
    fn wiki_links_with_url_encoding() {
        let content = "[[Page%20Name]]";
        assert_eq!(get_wiki_links(content), vec!["Page Name"]);
    }

    #[test]
    fn wiki_links_strip_md_extension() {
        let content = "[[Page.md]]";
        assert_eq!(get_wiki_links(content), vec!["Page"]);
    }

    #[test]
    fn wiki_links_invalid_encoding_keeps_original() {
        let content = "[[Page%ZZ]]";
        assert_eq!(get_wiki_links(content), vec!["Page%ZZ"]);
    }

    #[test]
    fn markdown_links_simple() {
        let content = "[text](Page)";
        assert_eq!(get_wiki_links(content), vec!["Page"]);
    }

    #[test]
    fn markdown_links_with_path() {
        let content = "[text](./sub/Page)";
        assert_eq!(get_wiki_links(content), vec!["./sub/Page"]);
    }

    #[test]
    fn markdown_links_external_skipped() {
        let content = "[text](https://example.com)";
        assert!(get_wiki_links(content).is_empty());
    }

    #[test]
    fn markdown_links_strip_md_extension() {
        let content = "[text](Page.md)";
        assert_eq!(get_wiki_links(content), vec!["Page"]);
    }

    #[test]
    fn markdown_links_with_url_encoding() {
        let content = "[text](Page%20Name)";
        assert_eq!(get_wiki_links(content), vec!["Page Name"]);
    }

    #[test]
    fn markdown_links_invalid_encoding_keeps_original() {
        let content = "[text](Page%ZZ)";
        assert_eq!(get_wiki_links(content), vec!["Page%ZZ"]);
    }

    #[test]
    fn mixed_wiki_and_markdown_links() {
        let content = "Start [[Wiki]] and [text](Markdown) end.";
        let links = get_wiki_links(content);
        assert_eq!(links, vec!["Wiki", "Markdown"]);
    }

    #[test]
    fn duplicate_links_are_not_deduplicated() {
        let content = "[[Page]] [[Page]]";
        assert_eq!(get_wiki_links(content), vec!["Page", "Page"]);
    }

    #[test]
    fn no_links_returns_empty() {
        let content = "Just plain text without any brackets.";
        assert!(get_wiki_links(content).is_empty());
    }

    #[test]
    fn empty_string_returns_empty() {
        assert!(get_wiki_links("").is_empty());
    }

    #[test]
    fn unclosed_wiki_link_does_not_match() {
        let content = "[[Unclosed";
        assert!(get_wiki_links(content).is_empty());
    }
}