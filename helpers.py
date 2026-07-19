"""
helpers.py — Shared constants, lookup maps, and prediction helpers.

Used by routers/ and app.py to avoid circular imports.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from fastapi import HTTPException

import db
import config
import data_collector
from train import build_features
from scrapers.utils import format_msk


# ── sstats league mapping ───────────────────────────────────────────────────

SSTATS_LEAGUE_ID = {
    "EPL": 39, "La_liga": 140, "Bundesliga": 78,
    "Serie_A": 135, "Ligue_1": 61, "RFPL": 235,
}


# ── League name translation ─────────────────────────────────────────────────

LEAGUE_NAME_MAP = {
    "World Cup": "Чемпионат мира",
    "Friendlies Clubs": "Товарищеские матчи",
    "Friendlies": "Товарищеские матчи",
    "Premier League": "АПЛ",
    "La Liga": "Ла Лига",
    "Bundesliga": "Бундеслига",
    "Serie A": "Серия А",
    "Ligue 1": "Лига 1",
    "Eredivisie": "Эредивизи",
    "Primeira Liga": "Примейра Лига",
    "Super Lig": "Супер Лига",
    "Championship": "Чемпионшип",
    "MLS": "MLS",
    "Champions League": "Лига чемпионов",
    "Europa League": "Лига Европы",
    "Conference League": "Конференц-лига",
    "Copa Libertadores": "Кубок Либертадорес",
    "Copa Sudamericana": "Кубок Южной Америки",
    "Brazil Serie A": "Бразилия. Серия А",
    "Argentina Liga Profesional": "Аргентина. Лига Про",
    "Mexico Liga MX": "Мексика. Лига MX",
    "Japan J1 League": "Япония. J1 Лига",
    "Japan J2 League": "Япония. J2 Лига",
    "Japan J3 League": "Япония. J3 Лига",
    "South Korea K League 1": "Южная Корея. K Лига 1",
    "South Korea K League 2": "Южная Корея. K Лига 2",
    "South Korea K3 League": "Южная Корея. K3 Лига",
    "South Korea K4 League": "Южная Корея. K4 Лига",
    "K3 League": "Южная Корея. K3 Лига",
    "Australia A-League": "Австралия. A-Лига",
    "China Super League": "Китай. Суперлига",
    "India Super League": "Индия. Суперлига",
    "Saudi Pro League": "Саудовская Аравия. Про-Лига",
    "La Liga 2": "Испания. Ла Лига 2",
    "2. Bundesliga": "Германия. 2 Бундеслига",
    "Serie B": "Италия. Серия Б",
    "Ligue 2": "Франция. Лига 2",
    "Kansallinen Liiga": "Финляндия. Высшая лига",
    "Meistriliiga": "Эстония. Высшая лига",
    "A Lyga": "Литва. A Лига",
    "Ykkösliiga": "Финляндия. Ykkösliiga",
    "Super Liga": "Молдова. Супер Лига",
    "Liga Pro": "Эквадор. Liga Pro",
    "Botola Pro": "Марокко. Botola Pro",
    "Canadian Premier League": "Канада. Премьер-лига",
    "Brasiliense U20": "Бразилия. U20",
    "Carioca C": "Бразилия. Кариока C",
    "Baiano - 2": "Бразилия. Байано 2",
    "Reserve League": "Аргентина. Резерв",
    "Torneo Federal A": "Аргентина. Federal A",
    "Torneo Promocional Amateur": "Аргентина. Promocional Amateur",
    "Primera Nacional": "Аргентина. Nacional",
    "Primera B Metropolitana": "Аргентина. B Metropolitana",
    "Primera C": "Аргентина. C",
    "Segunda División": "Чили. Segunda División",
    "Copa Chile": "Кубок Чили",
    "Division Intermedia": "Парагвай. Intermedia",
    "Northern Super League": "Канада. Женская лига",
    "Liga Women": "Женская лига",
    "MLS Next Pro": "США. MLS Next Pro",
    "USL League Two": "США. USL League Two",
    "Second League - Group 2": "Россия. Вторая Лига (Группа 2)",
    "Victoria NPL 2": "Австралия. Виктория NPL 2",
    "New South Wales NPL 2": "Австралия. Новый Южный Уэльс NPL 2",
    "NPL Victoria": "Австралия. NPL Виктория",
    "NPL New South Wales": "Австралия. NPL Новый Южный Уэльс",
    "Queensland NPL": "Австралия. NPL Квинсленд",
    "South Australia NPL": "Австралия. NPL Южная Австралия",
    "Western Australia NPL": "Австралия. NPL Западная Австралия",
    "1. Deild": "Исландия. 1 Дивизион",
    "2. Deild": "Исландия. 2 Дивизион",
    "Úrvalsdeild karla": "Исландия. Высшая лига",
    "Liga 1": "Перу. Лига 1",
    "Liga BetPlay": "Колумбия. Лига BetPlay",
    "Liga Profesional": "Боливия. Лига Про",
    "Liga FUTVE": "Венесуэла. Лига FUTVE",
    "Primera División": "Уругвай. Примера Дивизион",
    "Liga Paraguaya": "Парагвай. Лига",
    "Liga Nacional": "Гондурас. Национальная Лига",
    "Campeonato Nacional": "Куба. Национальный Чемпионат",
    "Liga Dominicana": "Доминикана. Лига",
    "Division 1": "Гаити. Дивизион 1",
    "TT Pro League": "Тринидад и Тобаго. Про-Лига",
    "SVB Eerste Divisie": "Суринам. Первый Дивизион",
    "GFF Elite League": "Гайана. Элитная Лига",
    "Premier League": "Премьер-лига",
}


# ── Team name translation (EN → RU) ────────────────────────────────────────

TEAM_NAME_MAP = {
    "Manchester United": "Манчестер Юнайтед",
    "Manchester City": "Манчестер Сити",
    "Liverpool": "Ливерпуль",
    "Arsenal": "Арсенал",
    "Chelsea": "Челси",
    "Tottenham": "Тоттенхэм",
    "Newcastle": "Ньюкасл",
    "Aston Villa": "Астон Вилла",
    "West Ham": "Уэст Хэм",
    "Brighton": "Брайтон",
    "Wolverhampton": "Вулверхэмптон",
    "Fulham": "Фулхэм",
    "Brentford": "Брентфорд",
    "Crystal Palace": "Кристал Пэлас",
    "Nottingham Forest": "Ноттингем Форест",
    "Bournemouth": "Борнмут",
    "Everton": "Эвертон",
    "Burnley": "Бернли",
    "Sheffield United": "Шеффилд Юнайтед",
    "Luton Town": "Лутон Таун",
    "Barcelona": "Барселона",
    "Real Madrid": "Реал Мадрид",
    "Atletico Madrid": "Атлетико Мадрид",
    "Sevilla": "Севилья",
    "Real Sociedad": "Реал Сосьедад",
    "Villarreal": "Вильярреал",
    "Athletic Bilbao": "Атлетик Бильбао",
    "Real Betis": "Реал Бетис",
    "Valencia": "Валенсия",
    "Girona": "Жирона",
    "Bayern Munich": "Бавария",
    "Borussia Dortmund": "Боруссия Дортмунд",
    "RB Leipzig": "РБ Лейпциг",
    "Bayer Leverkusen": "Байер Леверкузен",
    "Eintracht Frankfurt": "Айнтрахт Франкфурт",
    "VfB Stuttgart": "Штутгарт",
    "Wolfsburg": "Вольфсбург",
    "Freiburg": "Фрайбург",
    "Hoffenheim": " Хоффенхайм",
    "Union Berlin": "Юнион Берлин",
    "Inter Milan": "Интер",
    "AC Milan": "Милан",
    "Juventus": "Ювентус",
    "Napoli": "Наполи",
    "Roma": "Рома",
    "Lazio": "Лацио",
    "Atalanta": "Аталанта",
    "Fiorentina": "Фиорентина",
    "Torino": "Торино",
    "Bologna": "Болонья",
    "Paris Saint-Germain": "ПСЖ",
    "Lyon": "Лион",
    "Monaco": "Монако",
    "Marseille": "Марсель",
    "Lille": "Лилль",
    "Nice": "Ницца",
    "Rennes": "Ренн",
    "Lens": "Ланс",
    "Strasbourg": "Страсбур",
    "Montpellier": "Монпелье",
    "Ajax": "Аякс",
    "PSV": "ПСВ",
    "Feyenoord": "Фейеноорд",
    "AZ": "АЗ",
    "Twente": "Твенте",
    "Benfica": "Бенфика",
    "Porto": "Порту",
    "Sporting CP": "Спортинг",
    "Braga": "Брага",
    "Galatasaray": "Галатасарай",
    "Fenerbahce": "Фенербахче",
    "Besiktas": "Бешикташ",
    "Trabzonspor": "Трабзонспор",
    "Zenit": "Зенит",
    "Spartak Moscow": "Спартак",
    "CSKA Moscow": "ЦСКА",
    "Lokomotiv Moscow": "Локомотив",
    "Dynamo Moscow": "Динамо Москва",
    "Krasnodar": "Краснодар",
    "Rostov": "Ростов",
    "Club Brugge": "Брюгге",
    "Anderlecht": "Андерлехт",
    "Genk": "Генк",
    "Celtic": "Селтик",
    "Rangers": "Рейнджерс",
    "Red Bull Salzburg": "Зальцбург",
    "Sturm Graz": "Штурм",
    "Young Boys": "Базель",
    "FC Zurich": "Цюрих",
    "Olympiacos": "Олимпиакос",
    "Panathinaikos": "Панатинаикос",
    "PAOK": "ПАОК",
    "AEK Athens": "АЕК",
    "Slavia Prague": "Славия Прага",
    "Sparta Prague": "Спарта Прага",
    "Dinamo Zagreb": "Динамо Загреб",
    "Red Star Belgrade": "Црвена Звезда",
    "Shakhtar Donetsk": "Шахтёр Донецк",
    "Dynamo Kyiv": "Динамо Киев",
    "Brazil": "Бразилия", "Argentina": "Аргентина", "France": "Франция",
    "Germany": "Германия", "Spain": "Испания", "England": "Англия",
    "Italy": "Италия", "Portugal": "Португалия", "Netherlands": "Нидерланды",
    "Belgium": "Бельгия", "Croatia": "Хорватия", "Serbia": "Сербия",
    "Poland": "Польша", "Czech Republic": "Чехия", "Switzerland": "Швейцария",
    "Austria": "Австрия", "Denmark": "Дания", "Sweden": "Швеция",
    "Norway": "Норвегия", "Finland": "Финляндия", "Greece": "Греция",
    "Turkey": "Турция", "Ukraine": "Украина", "Russia": "Россия",
    "Scotland": "Шотландия", "Ireland": "Ирландия", "Wales": "Уэльс",
    "USA": "США", "Mexico": "Мексика", "Japan": "Япония",
    "South Korea": "Южная Корея", "Australia": "Австралия", "China": "Китай",
    "Iran": "Иран", "Saudi Arabia": "Саудовская Аравия",
    "Morocco": "Марокко", "Senegal": "Сенегал", "Cameroon": "Камерун",
    "Nigeria": "Нигерия", "Ghana": "Гана", "Tunisia": "Тунис",
    "Algeria": "Алжир", "Egypt": "Египет", "Canada": "Канада",
    "Colombia": "Колумбия", "Chile": "Чили", "Peru": "Перу",
    "Ecuador": "Эквадор", "Uruguay": "Уругвай", "Paraguay": "Парагвай",
    "Venezuela": "Венесуэла", "Bolivia": "Боливия",
    "Costa Rica": "Коста-Рика", "Panama": "Панама",
    "Honduras": "Гондурас", "Jamaica": "Ямайка",
    "Trinidad and Tobago": "Тринидад и Тобаго",
}


# ── Prediction helpers ──────────────────────────────────────────────────────

def model_predict_proba(rows: list, league_slug: str,
                        home_name: Optional[str] = None,
                        away_name: Optional[str] = None):
    """Works across model formats: v4/v3 (DC ensemble), v2 (XGB+LGB), v1 (single)."""
    from state import MODEL
    import pandas as pd
    import numpy as np
    fmt = MODEL.get("format", "v1")
    X = pd.DataFrame(rows, columns=MODEL["features"])
    # Convert None/NaN to proper float NaN for XGBoost
    X = X.astype(float).where(pd.notna(X), np.nan)
    if fmt in ("ensemble_v3", "ensemble_v4"):
        return MODEL["ensemble"].predict_proba(
            X, league_slug=league_slug,
            home_name=home_name, away_name=away_name,
        )
    if fmt == "ensemble_v2":
        return MODEL["ensemble"].predict_proba(X, league_slug=league_slug)
    return MODEL["model"].predict_proba(X)


def build_batch_features(home_id: int, away_id: int, home: dict, away: dict) -> list:
    """Build 19 features for batch-trained model."""
    import numpy as np
    WINDOW = 10
    today = dt.date.today()

    def _team_stats(tid, league_slug):
        with db.connect() as conn:
            matches = conn.execute(
                "SELECT * FROM matches WHERE is_result=1 "
                "AND (home_id=? OR away_id=?) ORDER BY date DESC LIMIT ?",
                (tid, tid, WINDOW)
            ).fetchall()
        if not matches:
            return [1.2, 1.2, 0.33, 0, 0, 0.33]
        gf, ga, wins, draws = 0, 0, 0, 0
        form = 0
        streak = 0
        for i, m in enumerate(matches):
            if m["home_id"] == tid:
                g, a = m["home_goals"] or 0, m["away_goals"] or 0
            else:
                g, a = m["away_goals"] or 0, m["home_goals"] or 0
            gf += g; ga += a
            if g > a: wins += 1; form += 3 * (len(matches) - i)
            elif g == a: draws += 1; form += 1 * (len(matches) - i)
        n = len(matches)
        for m in reversed(matches):
            if m["home_id"] == tid:
                g, a = m["home_goals"] or 0, m["away_goals"] or 0
            else:
                g, a = m["away_goals"] or 0, m["home_goals"] or 0
            if g > a:
                if streak >= 0: streak += 1
                else: break
            elif g < a:
                if streak <= 0: streak -= 1
                else: break
            else: break
        return [gf/n, ga/n, wins/n, form, streak, draws/n]

    hs = _team_stats(home_id, home["league_slug"])
    aws = _team_stats(away_id, away.get("league_slug", ""))
    h_elo = db.get_team_elo(home_id) or 1500
    a_elo = db.get_team_elo(away_id) or 1500
    h2h = db.head_to_head(home_id, away_id, limit=5) or []
    h2h_hw = sum(1 for m in h2h
                 if (m["home_id"] == home_id and (m["home_goals"] or 0) > (m["away_goals"] or 0))
                 or (m["away_id"] == home_id and (m["away_goals"] or 0) > (m["home_goals"] or 0)))

    def _rest(tid):
        with db.connect() as conn:
            m = conn.execute(
                "SELECT date FROM matches WHERE (home_id=? OR away_id=?) AND date<? ORDER BY date DESC LIMIT 1",
                (tid, tid, today.isoformat())
            ).fetchone()
        if m:
            try:
                d = dt.date.fromisoformat(m["date"][:10])
                return min((today - d).days, 30)
            except: pass
        return 7

    return [
        hs[0], hs[1], hs[2], hs[3], hs[4], hs[5],
        aws[0], aws[1], aws[2], aws[3], aws[4], aws[5],
        h_elo - a_elo, h2h_hw,
        0.33, 0.33, 0.33,
        _rest(home_id), _rest(away_id),
    ]


def predict_pair(home_id: int, away_id: int, home: dict, away: dict) -> dict:
    """Full prediction: build features → model predict → return probabilities."""
    from state import MODEL
    if MODEL is None:
        raise HTTPException(503, "Модель не загружена. Нажмите 'Обучить модель' или запустите train.py")

    features_count = len(MODEL.get("features", []))
    if features_count == 19:
        features = build_batch_features(home_id, away_id, home, away)
    else:
        with db.connect() as conn:
            prior = [dict(r) for r in conn.execute(
                "SELECT * FROM matches WHERE is_result=1 "
                "AND league_slug=? ORDER BY date DESC",
                (home["league_slug"],),
            ).fetchall()]
            # Find match_id for this pair (for market odds lookup)
            _match_row = conn.execute(
                "SELECT id FROM matches WHERE home_id=? AND away_id=? ORDER BY date DESC LIMIT 1",
                (home_id, away_id),
            ).fetchone()
            _match_id = _match_row[0] if _match_row else None
        today_iso = dt.date.today().isoformat()
        current_season = data_collector._current_season_year()
        features = build_features(
            home_id, away_id, prior,
            match_date=today_iso,
            league_slug=home["league_slug"],
            season=current_season,
            match_id=_match_id,
        )

    proba = model_predict_proba([features], home["league_slug"],
                                home_name=home["name"],
                                away_name=away["name"])[0]
    h2h = db.head_to_head(home_id, away_id, limit=5)

    return {
        "home": home,
        "away": away,
        "probabilities": {
            "home_win": round(float(proba[2]) * 100, 1),
            "draw":     round(float(proba[1]) * 100, 1),
            "away_win": round(float(proba[0]) * 100, 1),
        },
        "features": {name: (float(v) if isinstance(v, (int, float)) else v)
                     for name, v in zip(MODEL["features"], features)},
        "h2h_last5": h2h,
    }


def with_prediction(match: dict) -> dict:
    """Enrich a match dict with model prediction."""
    from state import MODEL
    out = dict(match)
    if match.get("date"):
        out["date_msk"] = format_msk(match["date"])
    if MODEL is not None:
        try:
            with db.connect() as conn:
                prior = [dict(r) for r in conn.execute(
                    "SELECT * FROM matches WHERE is_result=1 "
                    "AND league_slug=? AND date<? ORDER BY date DESC",
                    (match["league_slug"], match["date"]),
                ).fetchall()]
            features = build_features(
                match["home_id"], match["away_id"], prior,
                match_date=match["date"],
                league_slug=match["league_slug"],
                season=match["season"],
            )
            proba = model_predict_proba(
                [features], match["league_slug"],
                home_name=match.get("home_name"),
                away_name=match.get("away_name"))[0]
            out["our_prediction"] = {
                "home_win": round(float(proba[2]) * 100, 1),
                "draw":     round(float(proba[1]) * 100, 1),
                "away_win": round(float(proba[0]) * 100, 1),
            }
        except Exception:
            out["our_prediction"] = None
    return out
