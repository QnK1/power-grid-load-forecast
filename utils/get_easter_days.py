import pandas as pd


def get_easter_days_for_years(years: list[int]) -> dict[str, list[pd.Timestamp]]:
    easter_sunday = []
    good_friday = []
    easter_monday = []
    ascension_day = []
    pentecost_sunday = []
    whit_monday = []
    
    for year in years:
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        
        easter_sunday.append(pd.Timestamp(year=year, month=month, day=day))
        good_friday.append(easter_sunday[-1] - pd.Timedelta(days=2))
        easter_monday.append(easter_sunday[-1] + pd.Timedelta(days=1))
        ascension_day.append(easter_sunday[-1] + pd.Timedelta(days=39))
        pentecost_sunday.append(easter_sunday[-1] + pd.Timedelta(days=49))
        whit_monday.append(easter_sunday[-1] + pd.Timedelta(days=50))
    
    return {
        "Easter Sunday" : easter_sunday,
        "Good Friday" : good_friday,
        "Easter Monday" : easter_monday,
        "Ascension Day" : ascension_day,
        "Pentecost Sunday" : pentecost_sunday,
        "Whit Monday" : whit_monday,
    }