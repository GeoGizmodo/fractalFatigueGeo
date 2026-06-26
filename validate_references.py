#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference Validation Script
============================
Extracts all bibitem entries from the manuscript .tex file,
queries Crossref API for each title, and reports a confidence score.

Output: CSV audit report + console summary.
"""

import re
import requests
import difflib
import csv
import time
import sys

# =============================================================================
# 1. EXTRACT REFERENCES FROM TEX FILE
# =============================================================================

def extract_bibitems(tex_file):
    """Extract bibitem keys and their text content from a .tex file."""
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all bibitems: \bibitem{key}\n text until next \bibitem or \end{thebibliography}
    pattern = r'\\bibitem\{([^}]+)\}\s*\n(.*?)(?=\\bibitem|\n\\end\{thebibliography\})'
    matches = re.findall(pattern, content, re.DOTALL)
    
    refs = []
    for key, text in matches:
        # Clean up the text
        text = text.strip()
        
        # Nature-style refs: Author. Title. \textit{Journal} vol, pages (year).
        # Title is the plain text BETWEEN the author block and \textit{Journal}
        # Strategy: find text between first period after authors and \textit{
        title = ""
        
        # Try: extract text between ".) " or "al. " and "\textit{"
        title_match = re.search(r'(?:et al\.|al\.|[\d]\))\s*(.+?)\s*\\textit\{', text)
        if title_match:
            title = title_match.group(1).strip().rstrip('.')
        else:
            # Fallback: text between first ". " and "\textit{"
            title_match2 = re.search(r'\.\s+(.+?)\s*\\textit\{', text)
            if title_match2:
                title = title_match2.group(1).strip().rstrip('.')
            else:
                # Last fallback: use \textit content (might be book title)
                title_match3 = re.search(r'\\textit\{([^}]+)\}', text)
                if title_match3:
                    title = title_match3.group(1)
                else:
                    title = text.split('.')[0] if '.' in text else text[:100]
        
        # Clean LaTeX commands from title
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
        title = re.sub(r'[\\{}$]', '', title)
        title = title.strip()
        
        # Extract year if possible
        year_match = re.search(r'\((\d{4})\)', text)
        year = year_match.group(1) if year_match else "unknown"
        
        # Extract authors (first part before title)
        authors = text.split('\\textit')[0].strip() if '\\textit' in text else text[:50]
        authors = re.sub(r'[\\{}$]', '', authors)
        
        refs.append({
            'key': key,
            'title': title,
            'year': year,
            'authors': authors[:80],
            'raw': text[:200],
        })
    
    return refs


# =============================================================================
# 2. CROSSREF API LOOKUP
# =============================================================================

def search_crossref(title, timeout=15):
    """Query Crossref API for a title. Returns top 3 results."""
    url = "https://api.crossref.org/works"
    params = {
        "query.title": title,
        "rows": 3,
    }
    headers = {
        "User-Agent": "ReferenceValidator/1.0 (mailto:aneesh@geogizmodo.ai)"
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["items"]
    except Exception as e:
        return [{"error": str(e)}]


def search_openlibrary(title, timeout=10):
    """Query Open Library for a book title."""
    url = "https://openlibrary.org/search.json"
    params = {"title": title, "limit": 3}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        docs = r.json().get("docs", [])
        if not docs:
            return "BOOK-NOT-FOUND"
        best = docs[0]
        found_title = best.get("title", "")
        score = similarity(title, found_title)
        year = best.get("first_publish_year", "?")
        if score >= 0.6:
            return f"BOOK-VERIFIED (score={score:.2f}, year={year}, found: {found_title[:60]})"
        else:
            return f"BOOK-CHECK (score={score:.2f}, found: {found_title[:60]})"
    except Exception as e:
        return f"BOOK-ERROR ({e})"


def similarity(a, b):
    """Compute string similarity between two titles."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# =============================================================================
# 3. VALIDATE ALL REFERENCES
# =============================================================================

def validate_references(tex_file, output_csv="reference_audit.csv"):
    """Main validation pipeline."""
    
    print("=" * 70)
    print("REFERENCE VALIDATION AUDIT")
    print(f"Source: {tex_file}")
    print("=" * 70)
    
    # Extract references
    refs = extract_bibitems(tex_file)
    print(f"\nExtracted {len(refs)} references from {tex_file}")
    
    # Validate each
    results = []
    
    for i, ref in enumerate(refs):
        print(f"\n[{i+1}/{len(refs)}] {ref['key']}")
        print(f"  Title: {ref['title'][:70]}...")
        print(f"  Year:  {ref['year']}")
        
        # Skip books and standards (Crossref coverage is poor for these)
        is_book = any(kw in ref['raw'].lower() for kw in 
                     ['isbn', 'edition', 'press', 'publishers',
                      'freeman', 'wiley', 'springer', 'elsevier', 'academic press',
                      'dover', 'chapman', 'pearson', 'cambridge university'])
        is_standard = any(kw in ref['raw'].lower() for kw in 
                        ['iso ', 'astm ', 'standard'])
        
        if is_book:
            # Try Open Library for books
            print(f"  -> BOOK: checking Open Library...")
            book_status = search_openlibrary(ref['title'])
            print(f"     {book_status}")
            results.append({
                'key': ref['key'],
                'title': ref['title'],
                'year': ref['year'],
                'type': 'book',
                'confidence': book_status,
                'doi': 'N/A',
                'best_match': 'N/A',
                'status': book_status,
            })
            time.sleep(0.5)
            continue
        
        if is_standard:
            print(f"  -> STANDARD (skipping Crossref lookup)")
            results.append({
                'key': ref['key'],
                'title': ref['title'],
                'year': ref['year'],
                'type': 'standard',
                'confidence': 'N/A (standard)',
                'doi': 'N/A',
                'best_match': 'N/A',
                'status': 'SKIP-STANDARD',
            })
            continue
        
        # Query Crossref
        items = search_crossref(ref['title'])
        
        if not items or 'error' in items[0]:
            err = items[0].get('error', 'No results') if items else 'No results'
            print(f"  -> ERROR: {err}")
            results.append({
                'key': ref['key'],
                'title': ref['title'],
                'year': ref['year'],
                'type': 'article',
                'confidence': 0.0,
                'doi': 'ERROR',
                'best_match': str(err),
                'status': 'ERROR',
            })
            time.sleep(1)
            continue
        
        # Find best match
        best_score = 0
        best_item = None
        for item in items:
            found_title = item.get("title", [""])[0] if item.get("title") else ""
            score = similarity(ref['title'], found_title)
            if score > best_score:
                best_score = score
                best_item = item
        
        # Extract info from best match
        if best_item:
            found_title = best_item.get("title", [""])[0] if best_item.get("title") else ""
            doi = best_item.get("DOI", "No DOI")
            found_year = "?"
            for date_key in ["published-print", "published-online", "created"]:
                if date_key in best_item:
                    parts = best_item[date_key].get("date-parts", [["?"]])
                    if parts and parts[0]:
                        found_year = str(parts[0][0])
                        break
        else:
            found_title = "No match"
            doi = "No DOI"
            found_year = "?"
        
        # Determine status
        if best_score >= 0.85:
            status = "VERIFIED"
        elif best_score >= 0.60:
            status = "LIKELY-OK"
        elif best_score >= 0.40:
            status = "CHECK"
        else:
            status = "NOT-FOUND"
        
        # Year check
        if found_year != "?" and ref['year'] != "unknown":
            try:
                if abs(int(found_year) - int(ref['year'])) > 1:
                    status += " (YEAR-MISMATCH)"
            except ValueError:
                pass
        
        print(f"  -> {status} (score={best_score:.2f}, DOI={doi})")
        print(f"     Found: {found_title[:60]}")
        
        results.append({
            'key': ref['key'],
            'title': ref['title'],
            'year': ref['year'],
            'type': 'article',
            'confidence': f"{best_score:.3f}",
            'doi': doi,
            'best_match': found_title[:100],
            'status': status,
        })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'key', 'title', 'year', 'type', 'confidence', 'doi', 'best_match', 'status'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'=' * 70}")
    print(f"AUDIT COMPLETE. Results saved to: {output_csv}")
    print(f"{'=' * 70}")
    
    # Summary
    verified = sum(1 for r in results if 'VERIFIED' in r['status'])
    likely = sum(1 for r in results if 'LIKELY' in r['status'])
    check = sum(1 for r in results if r['status'] == 'CHECK')
    not_found = sum(1 for r in results if 'NOT-FOUND' in r['status'])
    skipped = sum(1 for r in results if 'SKIP' in r['status'])
    errors = sum(1 for r in results if 'ERROR' in r['status'])
    
    print(f"\nSUMMARY:")
    print(f"  VERIFIED (score >= 0.85): {verified}")
    print(f"  LIKELY-OK (0.60-0.85):   {likely}")
    print(f"  CHECK (0.40-0.60):       {check}")
    print(f"  NOT-FOUND (< 0.40):      {not_found}")
    print(f"  ERRORS:                  {errors}")
    print(f"  TOTAL:                   {len(results)}")
    
    # Print only items needing attention
    attention = [r for r in results if r['status'] not in ('VERIFIED', 'SKIP-STANDARD') 
                 and 'BOOK-VERIFIED' not in r['status']]
    
    if attention:
        print(f"\n{'=' * 70}")
        print(f"ITEMS NEEDING MANUAL CHECK ({len(attention)}):")
        print(f"{'=' * 70}")
        for r in attention:
            print(f"\n  [{r['status']}] {r['key']}")
            print(f"    Title: {r['title'][:70]}")
            print(f"    Year:  {r['year']}")
            if r['doi'] not in ('N/A', 'ERROR'):
                print(f"    DOI:   {r['doi']}")
            if r['best_match'] != 'N/A':
                print(f"    Match: {r['best_match'][:70]}")
    else:
        print(f"\n  All references verified or skipped (books/standards).")
    
    return results


# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == '__main__':
    tex_file = sys.argv[1] if len(sys.argv) > 1 else "nature_communications_manuscript_comments_addressed.tex"
    validate_references(tex_file)
