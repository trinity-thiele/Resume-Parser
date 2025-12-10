import json
from pathlib import Path
from typing import List, Dict, Any


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"File not found: {path}")
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load JSON {path}: {e}")
        return []


def extract_bow_top3(item: Dict[str, Any]) -> List[str]:
    # Expect item to have 'top_3_categories' or 'model_top3' or use 'bow_prediction'
    if 'top_3_categories' in item and item['top_3_categories']:
        return [str(x.get('category') or x.get('cluster') or x) for x in item['top_3_categories']]
    if 'model_top3' in item and item['model_top3']:
        return [str(x.get('name') or x.get('cluster') or x) for x in item['model_top3']]
    if 'bow_prediction' in item:
        return [str(item['bow_prediction'])]
    return []


def extract_spacy_top3(item: Dict[str, Any]) -> List[str]:
    # Spacy classifier stored earlier as list of results with 'classification' containing 'top_three_clusters'
    cls = item.get('classification') or {}
    top = cls.get('top_three_clusters') or cls.get('top3') or []
    if isinstance(top, list) and top:
        names = []
        for t in top:
            # t may be dict with 'name' or 'cluster'
            if isinstance(t, dict):
                names.append(str(t.get('name') or t.get('cluster')))
            else:
                names.append(str(t))
        return names
    # fallback: look for 'primary_cluster_name'
    if 'primary_cluster_name' in cls:
        return [str(cls.get('primary_cluster_name'))]
    return []


def compare_lists(a: List[str], b: List[str]) -> Dict[str, Any]:
    set_a = set(a)
    set_b = set(b)
    top1_match = (a[0] == b[0]) if a and b else False
    top3_overlap = len(set_a & set_b)
    return {
        'top1_match': top1_match,
        'top3_overlap': top3_overlap,
        'a_top3': a,
        'b_top3': b
    }


def main():
    print("\nComparing Bag of Words and SpaCy/Gemini model results...\n")
    repo_root = Path(__file__).resolve().parent
    bow_path = repo_root / 'bow_results.json'
    spacy_path = Path(__file__).resolve().parent.parent / 'models' / 'spacy_data' / 'classified_resumes.json'

    bow = load_json(bow_path)
    spacy = load_json(spacy_path)

    # Index spacy by file_name for quick lookup
    spacy_index = { (item.get('file_name') or item.get('file') or '').strip(): item for item in spacy }

    comparisons = []
    stats = {'total': 0, 'top1_matches': 0, 'total_top3_overlap': 0}

    for item in bow:
        fname = (item.get('file_name') or '').strip()
        bow_top3 = extract_bow_top3(item)
        sp_item = spacy_index.get(fname)
        sp_top3 = extract_spacy_top3(sp_item) if sp_item else []

        comp = compare_lists(bow_top3, sp_top3)
        comp_record = {
            'file_name': fname,
            'bow_top3': comp['a_top3'],
            'spacy_top3': comp['b_top3'],
            'top1_match': comp['top1_match'],
            'top3_overlap': comp['top3_overlap']
        }
        comparisons.append(comp_record)

        stats['total'] += 1
        if comp['top1_match']:
            stats['top1_matches'] += 1
        stats['total_top3_overlap'] += comp['top3_overlap']

    # Summary
    if stats['total'] > 0:
        print(f"Compared {stats['total']} resumes")
        print(f"Top-1 agreement: {stats['top1_matches']} ({stats['top1_matches']/stats['total']:.2%})")
        avg_top3_overlap = stats['total_top3_overlap'] / stats['total']
        print(f"Average top-3 overlap: {avg_top3_overlap:.2f} items")

    out_summary = repo_root / 'comparison_summary.csv'
    out_details = repo_root / 'comparison_details.json'

    # Write CSV summary
    try:
        import csv
        with out_summary.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=['file_name','bow_top1','spacy_top1','top1_match','top3_overlap'])
            writer.writeheader()
            for r in comparisons:
                writer.writerow({
                    'file_name': r['file_name'],
                    'bow_top1': r['bow_top3'][0] if r['bow_top3'] else '',
                    'spacy_top1': r['spacy_top3'][0] if r['spacy_top3'] else '',
                    'top1_match': r['top1_match'],
                    'top3_overlap': r['top3_overlap']
                })
        print(f"Saved CSV summary to: {out_summary}")
    except Exception as e:
        print(f"Failed to write CSV summary: {e}")

    # Write detailed JSON
    try:
        out_details.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"Saved detailed JSON to: {out_details}")
    except Exception as e:
        print(f"Failed to write details JSON: {e}")


if __name__ == '__main__':
    main()
