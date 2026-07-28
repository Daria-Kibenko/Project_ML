import os
import re
import sys
import time
import hashlib
import logging
from datetime import datetime, timezone
from dataclasses import dataclass

import feedparser
import requests
from bs4 import BeautifulSoup

try:
    from keywords import is_spam, is_priority, get_incident_category
except ImportError:
    def is_spam(text, threshold=2):
        _S = ["реклама", "скидка", "купить", "промокод", "конкурс", "выиграй", "распродажа", "гороскоп"]
        return sum(1 for w in _S if w in text.lower()) >= threshold


    def is_priority(text):
        _P = ["выброс", "разлив", "авария", "пожар", "загрязнение", "утечка",
              "химикат", "эвакуация", "жалоба", "протест", "отравление", "пострадав", "взрыв"]
        return any(w in text.lower() for w in _P)


    def get_incident_category(text):
        t = text.lower()
        if any(w in t for w in ["выброс", "разлив", "загрязнение"]): return "экологический"
        if any(w in t for w in ["пожар", "взрыв", "авария"]): return "промышленный"
        if any(w in t for w in ["жалоба", "протест", "митинг"]): return "жалоба"
        if any(w in t for w in ["отравление", "госпитализир"]): return "здоровье"
        return "не определён"

logger = logging.getLogger(__name__)
VK_TOKEN = os.environ.get("VK_TOKEN", "aa36d183aa36d183aa36d183f8a974ced1aaa36aa36d183c07e1a5511d9a8bd6b9ff82c")

HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# RSS - федеральные и региональные СМИ
RSS_FEEDS = [
    # Федеральные - добавлены по запросу
    "https://www.mk.ru/rss/index.xml",
    "https://rss.rambler.ru/news/",
    "https://ria.ru/export/rss2/archive/index.xml",
    "https://www.gazeta.ru/export/rss/first.xml",
    "https://rbc.ru/rss/news",
    "https://tass.ru/rss/v2.xml",
    "https://news.mail.ru/rss/main/",
    "https://meduza.io/rss/all",
    "https://www.kommersant.ru/RSS/main.xml",
    "https://rg.ru/xml/index.xml",
    # Региональные
    "https://74.ru/rss/",
    "https://161.ru/rss/",
    "https://www.sibkray.ru/rss/",
    "https://bellona.ru/feed/",
    # Экология - официальные
    "https://www.mnr.gov.ru/press/news/rss/",
    "https://rpn.gov.ru/rss.xml",
    # Экология - специализированные (добавлены из второго списка)
    "https://nia.eco/feed/",  # Национальное информационное агентство «Экология»
    "https://ria.ru/eco/",  # РИА - экология
    # Бизнес и компании (добавлены из второго списка)
    "https://bcs-express.ru/rss",
]

# VK группы
# Формат: (screen_name, owner_id, posts_count)
# posts_count - сколько последних постов брать (больше для активных каналов)
VK_GROUPS = [
    # Крупные новостные - берём больше постов
    ("postnews", -57536014, 40),
    ("ostorozhno_news", None, 40),
    ("ostorozhno_sobchak", None, 40),
    ("mash", None, 40),
    ("mash.moyka", None, 40),
    ("piter_map", None, 40),
    ("today78", None, 40),
    # Экология
    ("ecosociety", None, 50),  # новая группа
    ("ekosamara", -57804775, 40),
    ("ecolog_rf", -57536014, 40),
    ("greenpeaceru", -17509782, 40),
    ("wwf_ru", -22729239, 40),
    ("bellona_ru", -34630408, 40),
    ("ecoaktivist", -55667630, 40),
    ("industrial_ecology", -120983178, 40),
    # ЧП города
    ("chp_msk", -51241793, 30),
    ("chp_spb", -47406090, 30),
    ("chp_ekb", -54490517, 20),
    ("chp_nnovogorod", -65049807, 20),
    ("chp_kazan", -62079947, 20),
    ("spb_today", -37468416, 20),
    # Города (все из списка)
    ("spb.live", None, 40), ("m24", None, 20),
    ("regiontymen", None, 40), ("rznnews62", None, 20),
    ("anapa.media", None, 40), ("life_samara", None, 20),
    ("saransk_photo", None, 40), ("ria_nos", None, 20),
    ("inshd", None, 40), ("novosibka", None, 20),
    ("omsk_vk", None, 40), ("omsk_live", None, 20),
    ("zlo43", None, 40), ("kirovonline", None, 20),
    ("ufa", None, 40), ("myufa", None, 20),
    ("kznlife", None, 40), ("te_ekb", None, 20),
    ("ekb", None, 40), ("etorostov", None, 20),
    ("cityvrn", None, 40), ("typical_krd", None, 20),
    ("bez_cenznn", None, 40), ("novgorod_52", None, 20),
    ("astrakhan_online", None, 40), ("permseychas", None, 20),
    ("prmvk", None, 40), ("otkrytiiarkhangelsk", None, 20),
    ("kemerovo_adm", None, 40), ("typical_kmr", None, 20),
    ("podslushanovptz", None, 40), ("ptzgovorit", None, 20),
    ("gorod34", None, 40), ("vld_today", None, 20),
    ("myvdk", None, 40), ("newsvlru", None, 20),
    ("onlinevologda", None, 40), ("vologda", None, 20),
    ("penzanovosti", None, 40), ("penza", None, 20),
    ("barneos22", None, 40), ("murmanskgroup", None, 20),
    ("i.murmansk", None, 40), ("tip32", None, 20),
    ("region53", None, 40), ("club17699037", None, 20),
    ("irknim", None, 40), ("kamchatka_now", None, 20),
    ("plnpsk", None, 40), ("orenburg_vk", None, 20),
    ("yakutsk_news", None, 40), ("newsykt", None, 20),
    ("typical_chelyabinsk", None, 40), ("nashchelyabinsk", None, 20),
    ("typical_adler", None, 40), ("typical_xm", None, 20),
    ("khanty_mansiyskonline", None, 40), ("hearsalehard", None, 20),
    ("salekhard_adm", None, 40), ("cheboksary", None, 20),
    ("yuzhnosakhalinsk_official", None, 40), ("in_yalta", None, 20),
    ("yalta_gr0up", None, 40), ("yarbv", None, 20),
    ("vesu_u", None, 40), ("sobytia03", None, 20),
    ("ulonline", None, 40), ("overhear.tula", None, 20),
    ("tulazest", None, 40), ("moytuapse", None, 20),
    ("tuapse.news", None, 40), ("tomsk123", None, 20),
    ("tomsk_ru", None, 40), ("tlt_01", None, 20),
    ("typical_tobolsk", None, 40), ("vktver", None, 20),
    ("tambov_pvt", None, 40), ("tambov.life", None, 20),
    ("nach_surgut", None, 40), ("26stav", None, 20),
    ("newsbelgorod", None, 40), ("smolensk67", None, 20),
    ("official_smolensk", None, 40), ("in_simferopol", None, 20),
    ("svetlogorskonline39", None, 40), ("history_rybinsk", None, 20),
    ("overhear_rybinsk", None, 40), ("sluxrv", None, 20),
    ("sarafannoeradiokmv", None, 40), ("pyatigorsklife", None, 20),
    ("typical_noginsk", None, 40), ("noginskgorodok", None, 20),
    ("nabchel", None, 40), ("interesnaya_kostroma", None, 20),
    ("chp44", None, 40), ("kaluganews_com", None, 20),
    ("ivnvo", None, 40), ("ivanovoobl37", None, 20),
    ("nashi_essentuki", None, 40), ("essentuki_online26", None, 20),
    ("typical__grozny", None, 40), ("groznytv", None, 20),
    ("gatchina47", None, 40), ("vyzma", None, 20),
    ("vyazmaonline", None, 40), ("vyborgvk", None, 20),
    ("vyborg_interesting", None, 40), ("overhear33", None, 20),
]

# Поиск по СМИ
SCRAPE_SOURCES = [
    {
        "name": "Коммерсантъ ESG",
        "url": "https://www.kommersant.ru/rubric/131",
        "type": "standard",
        "article_selector": "article, .uho__item, .article-preview",
        "title_sel": "h2, h3, .uho__name",
        "text_sel": "p, .uho__text",
        "link_sel": "a[href]"
    },
    {
        "name": "Финам — компании",
        "url": "https://www.finam.ru/publications/section/companies/",
        "type": "standard",
        "article_selector": ".publication-item, article, .news-item",
        "title_sel": "h2, h3, .publication-title",
        "text_sel": "p, .publication-lead",
        "link_sel": "a[href]"
    },
    {
        "name": "BCS Express",
        "url": "https://bcs-express.ru/",
        "type": "standard",
        "article_selector": "article, .article-item, .news-block__item",
        "title_sel": "h2, h3",
        "text_sel": "p, .article-lead",
        "link_sel": "a[href]"
    },
    {
        "name": "ko.ru",
        "url": "https://ko.ru/",
        "type": "standard",
        "article_selector": "article, .article-card, .post-card",
        "title_sel": "h2, h3",
        "text_sel": "p, .article-lead",
        "link_sel": "a[href]"
    },
    {
        "name": "iz.ru — экология",
        "url": "https://iz.ru/tag/ekologiia",
        "type": "standard",
        "article_selector": "article, .card, .news-feed__item",
        "title_sel": "h2, h3, .card__title",
        "text_sel": "p, .card__lead",
        "link_sel": "a[href]"
    },
    {
        "name": "nia.eco",
        "url": "https://nia.eco/",
        "type": "standard",
        "article_selector": "article, .post, .news-item",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "link_sel": "a[href]"
    },
    {
        "name": "Interfax — экология",
        "url": "https://www.interfax.ru/search/?df=05.07.2021&dt=05.07.2026&sec=0&phrase=%D1%8D%D0%BA%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D1%8F",
        "type": "interfax"
    },
    {
        "name": "Interfax — компания",
        "url": "https://www.interfax.ru/search/?df=05.07.2021&dt=05.07.2026&sec=0&phrase=%D0%BA%D0%BE%D0%BC%D0%BF%D0%B0%D0%BD%D0%B8%D1%8F",
        "type": "interfax"
    },
    {
        "name": "Ведомости",
        "url": "https://www.vedomosti.ru/search?query=%D1%8D%D0%BA%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D1%8F",
        "type": "vedomosti"
    }
]


# Структура поста
@dataclass
class Post:
    text: str
    url: str
    source: str
    source_type: str
    published: datetime
    post_id: str
    priority: bool = False
    incident_category: str = "не определён"
    is_comment: bool = False  # True = комментарий, False = пост
    likes: int = 0  # реакции/лайки

    def to_dict(self):
        return {
            "text": self.text, "url": self.url, "source": self.source,
            "source_type": self.source_type,
            "published": self.published.isoformat(),
            "post_id": self.post_id, "priority": self.priority,
            "incident_category": self.incident_category,
            "is_comment": self.is_comment, "likes": self.likes,
        }


def _make_id(text, url):
    return hashlib.md5(f"{url}::{text[:100]}".encode()).hexdigest()


def _clean(text):
    if not text:
        return ""
    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _make_post(text, url, source, source_type, published, is_comment=False, likes=0):
    if is_spam(text) or len(text) < 20:
        return None
    return Post(
        text=text, url=url, source=source, source_type=source_type,
        published=published, post_id=_make_id(text, url),
        priority=is_priority(text),
        incident_category=get_incident_category(text),
        is_comment=is_comment, likes=likes,
    )


# RSS

def parse_rss(url, max_posts=30):
    posts = []
    try:
        resp = requests.get(url, headers=HEADERS_BROWSER, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        source_name = feed.feed.get("title", url.split("/")[2])
        for entry in feed.entries[:max_posts]:
            raw = entry.get("summary") or entry.get("description") or entry.get("title") or ""
            text = _clean(raw)
            title = _clean(entry.get("title", ""))
            if len(text) < 30 and title:
                text = title + ". " + text
            published = datetime.now(timezone.utc)
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
            post = _make_post(text, entry.get("link", url), source_name, "rss", published)
            if post: posts.append(post)
    except Exception as e:
        logger.debug(f"[rss] {url.split('/')[2]}: {e}")
    return posts


def parse_all_rss():
    all_posts = []
    for url in RSS_FEEDS:
        posts = parse_rss(url)
        all_posts.extend(posts)
        pri = sum(1 for p in posts if p.priority)
        logger.info(f"[rss] {'✅' if posts else '❌'} {url.split('/')[2]}: {len(posts)} постов, {pri} приор.")
        time.sleep(0.5)
    return all_posts


# Скрапинг СМИ

def scrape_source(name, url, art_sel, title_sel, text_sel, link_sel, max_posts=30):
    posts = []
    try:
        resp = requests.get(url, headers=HEADERS_BROWSER, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        items = []
        for sel in art_sel.split(","):
            items = soup.select(sel.strip())
            if items: break

        for item in items[:max_posts]:
            title_el = None
            for sel in title_sel.split(","):
                title_el = item.select_one(sel.strip())
                if title_el: break

            text_el = None
            for sel in text_sel.split(","):
                text_el = item.select_one(sel.strip())
                if text_el: break

            title = _clean(title_el.get_text()) if title_el else ""
            body = _clean(text_el.get_text()) if text_el else ""
            text = (title + ". " + body).strip() if body else title
            if not text or len(text) < 20: continue

            link_el = item.select_one(link_sel) if link_sel else None
            link = ""
            if link_el:
                href = link_el.get("href", "")
                link = href if href.startswith("http") else f"https://{url.split('/')[2]}{href}"

            post = _make_post(text, link or url, name, "scrape",
                              datetime.now(timezone.utc))
            if post: posts.append(post)

    except Exception as e:
        logger.debug(f"[scrape] {name}: {e}")
    return posts


def parse_all_scrape():
    all_posts = []
    for src in SCRAPE_SOURCES:
        posts = scrape_source(*src)
        all_posts.extend(posts)
        pri = sum(1 for p in posts if p.priority)
        logger.info(f"[scrape] {'✅' if posts else '❌'} {src[0]}: {len(posts)} постов, {pri} приор.")
        time.sleep(2)
    return all_posts


# Скрапинг СМИ

def scrape_source(name, url, art_sel, title_sel, text_sel, link_sel, max_posts=30):
    posts = []
    try:
        resp = requests.get(url, headers=HEADERS_BROWSER, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        items = []
        for sel in art_sel.split(","):
            items = soup.select(sel.strip())
            if items: break

        for item in items[:max_posts]:
            title_el = None
            for sel in title_sel.split(","):
                title_el = item.select_one(sel.strip())
                if title_el: break

            text_el = None
            for sel in text_sel.split(","):
                text_el = item.select_one(sel.strip())
                if text_el: break

            title = _clean(title_el.get_text()) if title_el else ""
            body = _clean(text_el.get_text()) if text_el else ""
            text = (title + ". " + body).strip() if body else title
            if not text or len(text) < 20: continue

            link_el = item.select_one(link_sel) if link_sel else None
            link = ""
            if link_el:
                href = link_el.get("href", "")
                link = href if href.startswith("http") else f"https://{url.split('/')[2]}{href}"

            post = _make_post(text, link or url, name, "scrape",
                              datetime.now(timezone.utc))
            if post: posts.append(post)

    except Exception as e:
        logger.debug(f"[scrape] {name}: {e}")
    return posts


def parse_all_scrape():
    all_posts = []
    for src in SCRAPE_SOURCES:
        posts = scrape_source(*src)
        all_posts.extend(posts)
        pri = sum(1 for p in posts if p.priority)
        logger.info(f"[scrape] {'✅' if posts else '❌'} {src[0]}: {len(posts)} постов, {pri} приор.")
        time.sleep(2)
    return all_posts


# VK

def resolve_vk_id(screen_name):
    if not VK_TOKEN: return None
    try:
        r = requests.get("https://api.vk.com/method/utils.resolveScreenName",
                         params={"screen_name": screen_name, "access_token": VK_TOKEN, "v": "5.199"},
                         timeout=10)
        d = r.json().get("response", {})
        if d.get("type") in ("group", "page", "club"):
            return -d["object_id"]
    except Exception:
        pass
    return None


def parse_vk_comments(owner_id, post_id_vk, post_url, source_name, min_likes=3):
    if not VK_TOKEN: return []
    comments = []
    try:
        r = requests.get("https://api.vk.com/method/wall.getComments",
                         params={"owner_id": owner_id, "post_id": post_id_vk,
                                 "count": 100, "sort": "desc", "need_likes": 1,
                                 "access_token": VK_TOKEN, "v": "5.199"}, timeout=10)
        for item in r.json().get("response", {}).get("items", []):
            if item.get("likes", {}).get("count", 0) < min_likes: continue
            text = _clean(item.get("text", ""))
            pub = datetime.fromtimestamp(item["date"], tz=timezone.utc)
            p = _make_post(text, f"{post_url}?reply={item['id']}",
                           f"{source_name}/комм", "vk_comment", pub,
                           is_comment=True, likes=item["likes"]["count"])
            if p: comments.append(p)
    except Exception:
        pass
    return comments


def parse_vk_group(screen_name, owner_id=None, count=30):
    if not VK_TOKEN: return []
    posts = []
    try:
        params = {"count": count, "filter": "owner",
                  "access_token": VK_TOKEN, "v": "5.199"}
        if owner_id:
            params["owner_id"] = owner_id
        else:
            resolved = resolve_vk_id(screen_name)
            if resolved:
                params["owner_id"] = resolved
                owner_id = resolved
            else:
                params["domain"] = screen_name

        r = requests.get("https://api.vk.com/method/wall.get", params=params, timeout=10)
        data = r.json()
        if "error" in data: return []

        for item in data.get("response", {}).get("items", []):
            text = _clean(item.get("text", ""))
            post_url = f"https://vk.com/wall{item['owner_id']}_{item['id']}"
            pub = datetime.fromtimestamp(item["date"], tz=timezone.utc)
            likes = item.get("likes", {}).get("count", 0)
            n_comm = item.get("comments", {}).get("count", 0)
            p = _make_post(text, post_url, f"vk:{screen_name}", "vk", pub, likes=likes)
            if p:
                posts.append(p)
                if p.priority or n_comm > 5:
                    time.sleep(0.3)
                    posts.extend(parse_vk_comments(
                        item["owner_id"], item["id"], post_url,
                        f"vk:{screen_name}", min_likes=3))
    except Exception as e:
        logger.debug(f"[vk] {screen_name}: {e}")
    return posts


def parse_all_vk():
    if not VK_TOKEN:
        logger.info("[vk] VK_TOKEN не задан — пропускаем")
        return []
    all_posts = []
    for cfg_item in VK_GROUPS:
        sn, oid, cnt = cfg_item
        posts = parse_vk_group(sn, oid, cnt)
        all_posts.extend(posts)
        main_p = [p for p in posts if not p.is_comment]
        comms = [p for p in posts if p.is_comment]
        pri = [p for p in posts if p.priority]
        if posts:
            logger.info(f"[vk] ✅ {sn}: {len(main_p)} постов, {len(comms)} комм., {len(pri)} приор.")
        time.sleep(0.35)
    return all_posts


# Главная

def fetch_all_sources():
    all_posts = []
    all_posts.extend(parse_all_rss())
    all_posts.extend(parse_all_vk())
    all_posts.extend(parse_all_scrape())

    seen, unique = set(), []
    for p in all_posts:
        if p.post_id not in seen:
            seen.add(p.post_id)
            unique.append(p)

    priority = [p for p in unique if p.priority]
    comments = [p for p in unique if p.is_comment]
    logger.info(f"[sources] итого: {len(unique)} | приор: {len(priority)} | комм: {len(comments)}")
    return unique


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    posts = fetch_all_sources()
    pri = [p for p in posts if p.priority]
    print(f"\n✅ Итого: {len(posts)} постов, приоритетных: {len(pri)}")
    for p in pri[:5]:
        print(f"  🔴 [{p.incident_category}] {p.source}: {p.text[:100]}...")
