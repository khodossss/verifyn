from .domain import check_domain_reputation
from .extractor import extract_article_content
from .search import check_if_old_news, search_fact_checkers, web_search

ALL_TOOLS = [
    web_search,
    search_fact_checkers,
    check_if_old_news,
    extract_article_content,
    check_domain_reputation,
]

__all__ = ["ALL_TOOLS"]
