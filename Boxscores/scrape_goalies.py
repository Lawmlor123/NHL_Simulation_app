"""
scrape_goalies.py - Scrape starting goalies from DailyFaceoff.com
Step 1: Explore the page structure
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

# ── Team name mapping ──────────────────────────────────────────────
TEAM_ABBREV = {
    'anaheim': 'ANA', 'ducks': 'ANA', 'anaheim ducks': 'ANA',
    'boston': 'BOS', 'bruins': 'BOS', 'boston bruins': 'BOS',
    'buffalo': 'BUF', 'sabres': 'BUF', 'buffalo sabres': 'BUF',
    'calgary': 'CGY', 'flames': 'CGY', 'calgary flames': 'CGY',
    'carolina': 'CAR', 'hurricanes': 'CAR', 'carolina hurricanes': 'CAR',
    'chicago': 'CHI', 'blackhawks': 'CHI', 'chicago blackhawks': 'CHI',
    'colorado': 'COL', 'avalanche': 'COL', 'colorado avalanche': 'COL',
    'columbus': 'CBJ', 'blue jackets': 'CBJ', 'columbus blue jackets': 'CBJ',
    'dallas': 'DAL', 'stars': 'DAL', 'dallas stars': 'DAL',
    'detroit': 'DET', 'red wings': 'DET', 'detroit red wings': 'DET',
    'edmonton': 'EDM', 'oilers': 'EDM', 'edmonton oilers': 'EDM',
    'florida': 'FLA', 'panthers': 'FLA', 'florida panthers': 'FLA',
    'los angeles': 'LAK', 'kings': 'LAK', 'los angeles kings': 'LAK',
    'minnesota': 'MIN', 'wild': 'MIN', 'minnesota wild': 'MIN',
    'montreal': 'MTL', 'canadiens': 'MTL', 'montreal canadiens': 'MTL',
    'nashville': 'NSH', 'predators': 'NSH', 'nashville predators': 'NSH',
    'new jersey': 'NJD', 'devils': 'NJD', 'new jersey devils': 'NJD',
    'new york islanders': 'NYI', 'islanders': 'NYI',
    'new york rangers': 'NYR', 'rangers': 'NYR',
    'ottawa': 'OTT', 'senators': 'OTT', 'ottawa senators': 'OTT',
    'philadelphia': 'PHI', 'flyers': 'PHI', 'philadelphia flyers': 'PHI',
    'pittsburgh': 'PIT', 'penguins': 'PIT', 'pittsburgh penguins': 'PIT',
    'san jose': 'SJS', 'sharks': 'SJS', 'san jose sharks': 'SJS',
    'seattle': 'SEA', 'kraken': 'SEA', 'seattle kraken': 'SEA',
    'st. louis': 'STL', 'st louis': 'STL', 'blues': 'STL', 'st. louis blues': 'STL',
    'tampa bay': 'TBL', 'lightning': 'TBL', 'tampa bay lightning': 'TBL',
    'toronto': 'TOR', 'maple leafs': 'TOR', 'toronto maple leafs': 'TOR',
    'utah': 'UTA', 'utah hockey club': 'UTA',
    'vancouver': 'VAN', 'canucks': 'VAN', 'vancouver canucks': 'VAN',
    'vegas': 'VGK', 'golden knights': 'VGK', 'vegas golden knights': 'VGK',
    'washington': 'WSH', 'capitals': 'WSH', 'washington capitals': 'WSH',
    'winnipeg': 'WPG', 'jets': 'WPG', 'winnipeg jets': 'WPG',
}


def scrape_starting_goalies():
    """Fetch DailyFaceoff starting goalies page and explore structure"""
    
    url = "https://www.dailyfaceoff.com/starting-goalies/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    print(f"Fetching {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Status code: {response.status_code}")
    print(f"Content length: {len(response.text)} characters")
    
    # ── Save raw HTML for debugging ────────────────────────────────
    debug_file = "dailyfaceoff_debug.html"
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Saved raw HTML to {debug_file}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print(f"\nPage title: {soup.title.string if soup.title else 'No title'}")
    
    # ── Search for relevant elements ───────────────────────────────
    print("\n" + "=" * 50)
    print("SEARCHING FOR GOALIE-RELATED ELEMENTS")
    print("=" * 50)
    
    # Search 1: Look for goalie-related CSS classes
    all_elements = soup.find_all(class_=True)
    relevant_classes = set()
    for elem in all_elements:
        for cls in elem.get('class', []):
            cls_lower = cls.lower()
            if any(kw in cls_lower for kw in ['goal', 'game', 'match', 'start', 'card', 'lineup', 'team']):
                relevant_classes.add(cls)
    
    if relevant_classes:
        print(f"\nRelevant CSS classes found ({len(relevant_classes)}):")
        for cls in sorted(relevant_classes):
            # Count how many elements use this class
            count = len(soup.find_all(class_=cls))
            print(f"  .{cls}  ({count} elements)")
    else:
        print("\nNo obviously relevant CSS classes found.")
    
    # Search 2: Look for goalie names in page text
    page_text = soup.get_text()
    print(f"\nTotal text length: {len(page_text)} chars")
    
    # Look for known goalie names as a test
    test_names = ['Swayman', 'Skinner', 'Vasilevskiy', 'Hellebuyck', 'Shesterkin']
    found_names = [name for name in test_names if name in page_text]
    print(f"\nKnown goalie names found in page text: {found_names}")
    
    if not found_names:
        print("\n⚠️  No goalie names found in raw HTML!")
        print("   This likely means the page uses JavaScript rendering.")
        print("   We'll need Selenium. Run:")
        print("     pip install selenium webdriver-manager")
        return None
    
    # Search 3: If names ARE found, try to find the structure
    print("\n" + "=" * 50)
    print("GOALIE NAMES FOUND - ANALYZING STRUCTURE")
    print("=" * 50)
    
    # Find elements containing goalie names
    for name in found_names[:2]:  # Check first 2
        print(f"\nSearching for '{name}'...")
        elements = soup.find_all(string=re.compile(name))
        for elem in elements[:3]:
            parent = elem.parent
            # Walk up the tree to find the container
            print(f"  Found in: <{parent.name} class='{parent.get('class', '')}'> ")
            
            # Go up 3-5 levels to find the game container
            for i in range(5):
                parent = parent.parent
                if parent and parent.name:
                    classes = parent.get('class', [])
                    print(f"    Level {i+1} up: <{parent.name} class='{classes}'> "
                          f"(text length: {len(parent.get_text()[:200])})")
    
    # Search 4: Look for any table or structured data
    tables = soup.find_all('table')
    if tables:
        print(f"\nFound {len(tables)} tables")
        for i, table in enumerate(tables[:3]):
            print(f"  Table {i}: {table.get('class', 'no class')} - "
                  f"{len(table.find_all('tr'))} rows")
    
    # Search 5: Look for JSON/script data
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and any(kw in script.string for kw in ['goalie', 'starter', 'matchup']):
            print(f"\n📌 Found script with goalie data!")
            print(f"   First 500 chars: {script.string[:500]}")
    
    # Search 6: Check for status indicators (confirmed, expected, etc.)
    for status in ['confirmed', 'expected', 'likely', 'unconfirmed']:
        elements = soup.find_all(string=re.compile(status, re.I))
        if elements:
            print(f"\n  Found '{status}' {len(elements)} times")
            # Show context around first occurrence
            first = elements[0]
            parent = first.parent
            for _ in range(3):
                parent = parent.parent
                if parent:
                    text = parent.get_text(separator=' | ', strip=True)[:300]
                    if name_found := [n for n in test_names if n in text]:
                        print(f"    Context: {text}")
                        break
    
    return page_text


if __name__ == "__main__":
    print("=" * 60)
    print("  DailyFaceoff Starting Goalies - Explorer")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    result = scrape_starting_goalies()
    
    if result is None:
        print("\n" + "=" * 60)
        print("NEXT STEP: We need Selenium for JavaScript rendering")
        print("Run: pip install selenium webdriver-manager")
        print("Then we'll build scrape_goalies_v2.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("NEXT STEP: Paste the output above and I'll build")
        print("the actual parser based on the HTML structure!")
        print("=" * 60)