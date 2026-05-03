from datetime import date
from collect_range import collect_range  # if you moved the function elsewhere

if __name__ == "__main__":
    collect_range(date(2025, 10, 8), date.today())