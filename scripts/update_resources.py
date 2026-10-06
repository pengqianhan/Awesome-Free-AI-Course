#!/usr/bin/env python3
"""扫描 GitHub 上 star 数超过 1000 的 AI 学习类项目，按类别追加到 resources/deeplearning_material.md。

流程：
1. 用 CATEGORIES 里的搜索词在 GitHub 搜索，收集候选项目；
2. 只保留名称或简介里像「学习资料」的项目（教程、课程、笔记、书……）；
3. 用每个类别的关键词给项目打分：名称命中 3 分、简介命中 2 分、topics 命中 1 分，
   归入得分最高的类别，同分时取 CATEGORIES 里靠前的；名称和简介都没命中的项目跳过；
4. 类别在文档里已存在（`## 标题` 完全一致），就追加到该类别下的 `### GitHub 高星项目`；
   不存在则新建类别，放在「在线工具」之前，并加入目录。

本地试运行（不修改文件）：
    GITHUB_TOKEN=$(gh auth token) python3 scripts/update_resources.py --dry-run
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "resources" / "deeplearning_material.md"
IGNORE_PATH = ROOT / ".github" / "resource-scan-ignore.txt"

# ---------------------------------------------------------------------------
# 可调整的配置
# ---------------------------------------------------------------------------

MIN_STARS = 1000            # star 数下限（不含）
MAX_PER_CATEGORY = 5        # 每次每个类别最多新增几个
MAX_TOTAL = 25              # 每次最多新增几个
ACTIVE_WITHIN_DAYS = 730    # 只收录最近两年内还有提交的项目
SUBSECTION = "### GitHub 高星项目"
NEW_CATEGORY_BEFORE = "在线工具"  # 新类别插在这个类别之前；找不到则放到文末

# title:    类别标题。和文档里的 `## 标题` 完全一致时归入已有类别，否则新建类别。
# queries:  GitHub 搜索词，只用来收集候选项目。
# keywords: 判断项目属于哪个类别的正则（匹配时已转小写，- 和 _ 已换成空格）。
# fallback: 为 True 时，只有其他类别都没命中才会归入这个类别。
# 列表顺序即同分时的优先级，越具体的类别越靠前。
CATEGORIES = [
    {
        "title": "扩散模型与流模型",
        "queries": ["diffusion model tutorial", "diffusion models course", "flow matching tutorial"],
        "keywords": r"diffusion|flow matching|score based generative|扩散模型",
    },
    {
        "title": "强化学习资料",
        "queries": ["reinforcement learning tutorial", "reinforcement learning course"],
        "keywords": r"reinforcement learning|\bdeep rl\b|\brl\b|强化学习",
    },
    {
        "title": "图神经网络资料",
        "queries": ["graph neural network tutorial", "gnn tutorial"],
        "keywords": r"graph neural|\bgnns?\b|graph representation|graph machine learning|图神经网络",
    },
    {
        "title": "AI Agent",
        "queries": ["ai agents course", "llm agents tutorial"],
        "keywords": r"\bagents?\b|agentic|智能体",
    },
    {
        "title": "LLM 学习资料",
        "queries": ["llm tutorial", "llm course", "large language model tutorial", "llm from scratch"],
        "keywords": r"\bllms?\b|large language models?|\bgpt\b|chatgpt|\brag\b|retrieval augmented|大模型|大语言模型",
    },
    {
        "title": "自然语言处理",
        "queries": ["nlp tutorial", "natural language processing course"],
        "keywords": r"\bnlp\b|natural language processing|自然语言处理",
    },
    {
        "title": "计算机视觉",
        "queries": ["computer vision tutorial", "computer vision course"],
        "keywords": r"computer vision|object detection|image segmentation|\byolo|计算机视觉",
    },
    {
        "title": "数学基础",
        "queries": ["mathematics for machine learning", "math for deep learning"],
        "keywords": r"mathematics|\bmaths?\b|linear algebra|calculus|probability|数学",
    },
    {
        "title": "深度学习资料",
        "queries": ["deep learning tutorial", "deep learning course", "deep learning book", "pytorch tutorial"],
        "keywords": r"deep learning|neural networks?|pytorch|tensorflow|深度学习",
    },
    {
        "title": "机器学习资料",
        "queries": ["machine learning tutorial", "machine learning course", "machine learning notes"],
        "keywords": r"machine learning|\bml\b|scikit learn|sklearn|机器学习",
    },
    {
        "title": "Python 学习资料",
        "queries": ["python tutorial beginners", "learn python course"],
        "keywords": r"\bpython\b",
        "fallback": True,
    },
]

# 名称或简介里出现这些词，才算「学习资料」，用来过滤掉框架、工具和应用类项目。
LEARNING_PATTERN = re.compile(
    r"\b(tutorials?|courses?|lectures?|notes|books?|handbook|guides?|learn|cookbook|"
    r"roadmap|curriculum|from scratch|hands on)\b|教程|课程|笔记|入门|手册",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------

GITHUB_REPO_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)


def search_repos(query, token):
    params = urllib.parse.urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": 30})
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "awesome-free-ai-course-resource-scan",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/search/repositories?{params}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["items"]
    except urllib.error.HTTPError as error:
        print(f"::warning::search failed ({error.code}) for query: {query}", file=sys.stderr)
        return []


def listed_repos(text):
    return {f"{owner}/{repo}".lower().removesuffix(".git") for owner, repo in GITHUB_REPO_RE.findall(text)}


def ignored_repos():
    if not IGNORE_PATH.exists():
        return set()
    entries = (line.split("#")[0].strip().lower() for line in IGNORE_PATH.read_text(encoding="utf-8").splitlines())
    return {entry for entry in entries if entry}


def normalize(text):
    return re.sub(r"[-_]", " ", text or "").lower()


def is_learning_resource(repo):
    return bool(LEARNING_PATTERN.search(normalize(f"{repo['name']} {repo.get('description')}")))


def classify(repo):
    """返回得分最高的类别标题；名称和简介都没命中任何类别时返回 None（只靠 topics 命中不可靠）。"""
    name = normalize(repo["name"])
    description = normalize(repo.get("description"))
    topics = normalize(" ".join(repo.get("topics", [])))

    scores = []
    for category in CATEGORIES:
        pattern = re.compile(category["keywords"])
        score = 3 * bool(pattern.search(name)) + 2 * bool(pattern.search(description)) + bool(pattern.search(topics))
        scores.append((score, category))

    matched = [(score, c) for score, c in scores if score >= 2]
    candidates = [(score, c) for score, c in matched if not c.get("fallback")] or matched
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    return next(c["title"] for score, c in candidates if score == best)


def format_item(repo):
    description = re.sub(r"\s+", " ", repo.get("description") or "").strip()
    if len(description) > 150:
        description = description[:149].rstrip(" .…") + "…"
    item = f"- [{repo['name']}]({repo['html_url']})"
    return f"{item}：{description}" if description else item


def slugify(title):
    """GitHub 标题锚点：小写，去掉标点，空格换成连字符。"""
    return re.sub(r"[^\w\- ]", "", title.strip().lower()).replace(" ", "-")


def find_section(lines, title):
    """返回 `## title` 小节的 (起始行, 结束行)，结束行为下一个 `## ` 标题，不包含在内。"""
    start = next((i for i, line in enumerate(lines) if line.strip() == f"## {title}"), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return start, end


def add_to_existing_category(lines, start, end, items):
    sub = next((i for i in range(start + 1, end) if lines[i].strip() == SUBSECTION), None)
    if sub is not None:
        insert_at = next((i for i in range(sub + 1, end) if lines[i].startswith("#")), end)
        while insert_at > sub + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines[insert_at:insert_at] = items
    else:
        insert_at = end
        while insert_at > start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines[insert_at:insert_at] = ["", SUBSECTION, "", *items]


def add_new_category(lines, title, items):
    block = [f"## {title}", "", SUBSECTION, "", *items, ""]
    before = find_section(lines, NEW_CATEGORY_BEFORE)
    if before:
        lines[before[0]:before[0]] = block
    else:
        while lines and not lines[-1].strip():
            lines.pop()
        lines.extend(["", *block])

    toc = find_section(lines, "目录")
    if toc:
        toc_items = [i for i in range(*toc) if lines[i].startswith("- [")]
        before_entry = next((i for i in toc_items if f"(#{slugify(NEW_CATEGORY_BEFORE)})" in lines[i]), None)
        position = before_entry if before_entry is not None else (toc_items[-1] + 1 if toc_items else toc[0] + 2)
        lines.insert(position, f"- [{title}](#{slugify(title)})")


def write_summary(path, picked, new_titles):
    total = sum(len(repos) for repos in picked.values())
    parts = [
        f"本周自动扫描到 **{total}** 个 star 超过 {MIN_STARS} 的学习类项目，已按类别加入 "
        "`resources/deeplearning_material.md`。",
        "",
    ]
    for title, repos in picked.items():
        parts.append(f"### {title}{'（新建类别）' if title in new_titles else ''}")
        parts.append("")
        for repo in repos:
            description = re.sub(r"\s+", " ", repo.get("description") or "").strip()
            parts.append(f"- [{repo['full_name']}]({repo['html_url']}) ⭐ {repo['stargazers_count']:,} — {description}")
        parts.append("")
    parts.append(
        "**审核方式**：不想收录的项目，在这个 PR 里删掉对应的行，并把 `owner/repo` 加到 "
        "`.github/resource-scan-ignore.txt`，以后的扫描就不会再出现它。"
    )
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def collect_candidates(token, seen):
    delay = 2.5 if token else 7  # 搜索接口限流：有 token 每分钟 30 次，没有每分钟 10 次
    since = (datetime.date.today() - datetime.timedelta(days=ACTIVE_WITHIN_DAYS)).isoformat()
    candidates = {}
    for category in CATEGORIES:
        for query in category["queries"]:
            full_query = (
                f"{query} in:name,description,topics stars:>{MIN_STARS} "
                f"fork:false archived:false pushed:>{since}"
            )
            for repo in search_repos(full_query, token):
                key = repo["full_name"].lower()
                if key not in seen and is_learning_resource(repo):
                    candidates[key] = repo
            time.sleep(delay)
    return sorted(candidates.values(), key=lambda repo: repo["stargazers_count"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不修改文件")
    parser.add_argument("--summary", help="把 PR 描述写到这个文件")
    args = parser.parse_args()

    text = MD_PATH.read_text(encoding="utf-8")
    seen = listed_repos(text) | ignored_repos()
    if os.environ.get("GITHUB_REPOSITORY"):
        seen.add(os.environ["GITHUB_REPOSITORY"].lower())

    # 按 CATEGORIES 的顺序排列，生成的文档和 PR 描述顺序稳定
    picked = {category["title"]: [] for category in CATEGORIES}
    total = 0
    for repo in collect_candidates(os.environ.get("GITHUB_TOKEN"), seen):
        title = classify(repo)
        if title is None or len(picked[title]) >= MAX_PER_CATEGORY:
            continue
        picked[title].append(repo)
        total += 1
        if total >= MAX_TOTAL:
            break
    picked = {title: repos for title, repos in picked.items() if repos}

    lines = text.split("\n")
    new_titles = set()
    for title, repos in picked.items():
        items = [format_item(repo) for repo in repos]
        section = find_section(lines, title)
        if section:
            add_to_existing_category(lines, *section, items)
        else:
            add_new_category(lines, title, items)
            new_titles.add(title)

    for title, repos in picked.items():
        print(f"{title}{' (new)' if title in new_titles else ''}:")
        for repo in repos:
            print(f"  {repo['full_name']}  ⭐ {repo['stargazers_count']}  {(repo.get('description') or '')[:70]}")
    print(f"Total new repos: {total}")

    if not args.dry_run and total:
        MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    if args.summary and total:
        write_summary(args.summary, picked, new_titles)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"count={total}\n")


if __name__ == "__main__":
    main()
