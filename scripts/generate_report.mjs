import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const API_BASE = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions";
const MODELS = ["GLM-5-Turbo", "GLM-4.7", "GLM-4.7-Flash"];
const MAX_RETRIES = 3;

function sanitizeHtml(str) {
  if (typeof str !== "string") return String(str);
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function extractJson(text) {
  if (!text || typeof text !== "string") return null;
  let cleaned = text.trim();
  const fenceMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) cleaned = fenceMatch[1].trim();
  const braceStart = cleaned.indexOf("{");
  const braceEnd = cleaned.lastIndexOf("}");
  if (braceStart !== -1 && braceEnd > braceStart) {
    cleaned = cleaned.slice(braceStart, braceEnd + 1);
  }
  try { return JSON.parse(cleaned); } catch {}
  cleaned = cleaned.replace(/,\s*([}\]])/g, "$1");
  cleaned = cleaned.replace(/\\(?!["\\/bfnrtu])/g, "\\\\");
  cleaned = cleaned.replace(/[\x00-\x1f]/g, (c) => {
    const code = c.charCodeAt(0);
    return code === 10 ? "\\n" : code === 13 ? "\\r" : code === 9 ? "\\t" : "";
  });
  try { return JSON.parse(cleaned); } catch {}
  const pairs = [];
  let depth = 0;
  for (let i = 0; i < cleaned.length; i++) {
    if (cleaned[i] === "{") { if (depth === 0) pairs.push(i); depth++; }
    if (cleaned[i] === "}") { depth--; if (depth === 0) pairs.push(i); }
  }
  for (let i = 0; i < pairs.length - 1; i += 2) {
    try { return JSON.parse(cleaned.slice(pairs[i], pairs[i + 1] + 1)); } catch {}
  }
  return null;
}

function buildPrompt(papers) {
  const paperList = papers.map((p, i) =>
    `[${i + 1}] PMID: ${p.pmid}\nTitle: ${p.title}\nJournal: ${p.journal}\nDate: ${p.date}\nAuthors: ${(p.authors || []).slice(0, 5).join(", ")}\nURL: ${p.url}`
  ).join("\n\n");

  return `你是薩提爾模式（Satir Model）研究文獻分析專家。以下是從 PubMed 檢索到的最新 Satir Model 相關研究文獻。

請進行以下分析：

1. 篩選出最值得關注的 5-8 篇論文（TOP Picks），其餘放入 other_papers
2. 為每篇論文生成：
   - 原始標題
   - 中文標題翻譯
   - 200-300 字中文摘要（基於標題推斷研究方向）
   - 薩提爾模式相關性分析
   - 研究設計類型
   - 臨床應用價值評分（1-5 星）
   - 分類標籤（從下方選項中選擇）
3. 生成整體文獻趨勢分析和關鍵詞

分類標籤：家族治療、團體諮商、伴侶治療、青少年心理健康、親子關係、創傷與受虐、成癮行為、自我價值、溝通姿態、一致性、情緒調節、靈性與意義、諮商師培訓、家庭功能、系統治療、經驗性治療、家庭重塑、冰山模式

請嚴格以 JSON 格式回應（不要加 markdown code block）：
{
  "summary": { "zh": "200字整體中文摘要", "en": "Brief English summary" },
  "trends": ["趨勢1", "趨勢2", "趨勢3"],
  "top_picks": [
    {
      "rank": 1,
      "title": "original title",
      "title_zh": "中文標題",
      "journal": "journal name",
      "date": "date",
      "url": "URL",
      "pmid": "PMID",
      "abstract_zh": "200-300字中文摘要",
      "satir_relevance": "薩提爾相關性分析",
      "study_type": "研究設計",
      "clinical_rating": 4,
      "tags": ["標籤1", "標籤2"],
      "keywords": ["keyword1", "keyword2"]
    }
  ],
  "other_papers": [
    {
      "title": "...",
      "title_zh": "...",
      "journal": "...",
      "date": "...",
      "url": "...",
      "pmid": "...",
      "abstract_zh": "...",
      "tags": ["..."]
    }
  ],
  "topic_distribution": { "家族治療": 3, "團體諮商": 2 },
  "hot_keywords": ["keyword1", "keyword2"]
}

文獻列表：
${paperList}`;
}

async function callZhipuAI(prompt, modelIndex = 0) {
  if (modelIndex >= MODELS.length) return null;
  const model = MODELS[modelIndex];
  const maxTokens = parseInt(process.env.MAX_TOKENS || "50000", 10);
  const timeoutMs = parseInt(process.env.TIMEOUT_MS || "480000", 10);
  const apiKey = process.env.ZHIPU_API_KEY;
  if (!apiKey) throw new Error("ZHIPU_API_KEY is not set");

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      console.log(`Calling ${model} (attempt ${attempt}/${MAX_RETRIES})...`);
      const res = await fetch(API_BASE, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: "你是專業的薩提爾模式研究文獻分析師。請用 JSON 格式回應。" },
            { role: "user", content: prompt },
          ],
          max_tokens: maxTokens,
          temperature: 0.7,
        }),
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (res.status === 429) {
        const wait = Math.min(5000 * attempt, 30000);
        console.log(`Rate limited, waiting ${wait}ms...`);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        console.error(`${model} HTTP ${res.status}: ${body.slice(0, 200)}`);
        break;
      }

      const data = await res.json();
      const content = data?.choices?.[0]?.message?.content;
      if (!content) {
        console.error(`${model} returned empty content`);
        break;
      }
      const parsed = extractJson(content);
      if (parsed && (parsed.top_picks || parsed.other_papers || parsed.summary)) {
        console.log(`${model} succeeded on attempt ${attempt}`);
        return parsed;
      }
      console.error(`${model} returned unparseable JSON, retrying...`);
    } catch (err) {
      console.error(`${model} attempt ${attempt} error: ${err.message}`);
      if (err.name === "AbortError" || err.name === "TimeoutError") break;
    }
  }
  console.log(`Falling back from ${model} to next model...`);
  return callZhipuAI(prompt, modelIndex + 1);
}

function generateFallbackReport(papers, targetDate) {
  return {
    summary: { zh: `本日從 PubMed 檢索到 ${papers.length} 篇薩提爾模式相關新文獻。AI 分析暫時不可用，以下為文獻基本資訊。`, en: `Found ${papers.length} new Satir Model papers from PubMed. AI analysis unavailable.` },
    trends: [],
    top_picks: papers.slice(0, 5).map((p, i) => ({
      rank: i + 1,
      title: p.title,
      title_zh: "",
      journal: p.journal,
      date: p.date,
      url: p.url,
      pmid: p.pmid,
      abstract_zh: "",
      satir_relevance: "",
      study_type: "",
      clinical_rating: 0,
      tags: [],
      keywords: [],
    })),
    other_papers: papers.slice(5).map((p) => ({
      title: p.title,
      title_zh: "",
      journal: p.journal,
      date: p.date,
      url: p.url,
      pmid: p.pmid,
      abstract_zh: "",
      tags: [],
    })),
    topic_distribution: {},
    hot_keywords: [],
  };
}

function escapeForTemplate(str) {
  if (typeof str !== "string") return "";
  return str.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$");
}

function generateHtml(report, targetDate) {
  const dayNames = ["日", "一", "二", "三", "四", "五", "六"];
  const dateObj = new Date(targetDate + "T00:00:00+08:00");
  const weekday = dayNames[dateObj.getDay()] || "";
  const dateDisplay = `${targetDate}（星期${weekday}）`;

  const starsHtml = (n) => {
    if (!n || n <= 0) return '<span style="color:var(--muted);font-size:0.85em">未評分</span>';
    return "★".repeat(n) + "☆".repeat(5 - n);
  };

  const tagHtml = (tags) => (tags || []).map((t) =>
    `<span style="display:inline-block;padding:3px 12px;border-radius:999px;font-size:0.78em;background:var(--accent-soft);color:var(--accent);margin:3px 4px 3px 0">${sanitizeHtml(t)}</span>`
  ).join("");

  const topCards = (report.top_picks || []).map((p) => `
    <div style="background:var(--card-bg);border-radius:24px;padding:28px 32px;border-left:3px solid var(--accent);box-shadow:0 2px 12px rgba(139,79,43,0.07);transition:transform .2s,box-shadow .2s;margin-bottom:20px"
         onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 24px rgba(139,79,43,0.13)'"
         onmouseout="this.style.transform='';this.style.boxShadow='0 2px 12px rgba(139,79,43,0.07)'">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <span style="background:var(--accent);color:#fff;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9em;flex-shrink:0">${p.rank || "#"}</span>
        <span style="font-size:0.82em;color:var(--muted)">${sanitizeHtml(p.journal || "")}</span>
        <span style="font-size:0.78em;color:var(--muted)">${sanitizeHtml(p.date || "")}</span>
      </div>
      <h3 style="margin:0 0 4px;font-size:1.05em;line-height:1.5;color:var(--text)">
        <a href="${sanitizeHtml(p.url || "#")}" target="_blank" rel="noopener noreferrer" style="color:var(--text);text-decoration:none">${sanitizeHtml(p.title || "")}</a>
      </h3>
      ${p.title_zh ? `<p style="margin:0 0 10px;font-size:0.9em;color:var(--accent);font-weight:500">${sanitizeHtml(p.title_zh)}</p>` : ""}
      ${p.abstract_zh ? `<p style="margin:0 0 12px;font-size:0.88em;line-height:1.7;color:var(--text)">${sanitizeHtml(p.abstract_zh)}</p>` : ""}
      ${p.satir_relevance ? `<div style="background:var(--surface);border-radius:12px;padding:12px 16px;margin-bottom:10px;font-size:0.84em;line-height:1.6"><strong style="color:var(--accent)">薩提爾相關性：</strong>${sanitizeHtml(p.satir_relevance)}</div>` : ""}
      ${p.study_type ? `<div style="font-size:0.82em;color:var(--muted);margin-bottom:6px"><strong>研究設計：</strong>${sanitizeHtml(p.study_type)}</div>` : ""}
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap">
        ${p.clinical_rating ? `<span style="font-size:0.9em;color:#d4922a;letter-spacing:2px">${starsHtml(p.clinical_rating)}</span>` : ""}
      </div>
      ${tagHtml(p.tags)}
    </div>`).join("");

  const otherCards = (report.other_papers || []).map((p) => `
    <div style="background:var(--card-bg);border-radius:16px;padding:18px 22px;box-shadow:0 1px 6px rgba(139,79,43,0.05);margin-bottom:12px">
      <h4 style="margin:0 0 4px;font-size:0.92em">
        <a href="${sanitizeHtml(p.url || "#")}" target="_blank" rel="noopener noreferrer" style="color:var(--text);text-decoration:none">${sanitizeHtml(p.title || "")}</a>
      </h4>
      ${p.title_zh ? `<p style="margin:0 0 6px;font-size:0.84em;color:var(--accent)">${sanitizeHtml(p.title_zh)}</p>` : ""}
      <div style="font-size:0.8em;color:var(--muted);margin-bottom:6px">${sanitizeHtml(p.journal || "")} · ${sanitizeHtml(p.date || "")}</div>
      ${p.abstract_zh ? `<p style="margin:0 0 8px;font-size:0.84em;line-height:1.6;color:var(--text)">${sanitizeHtml(p.abstract_zh)}</p>` : ""}
      ${tagHtml(p.tags)}
    </div>`).join("");

  const distEntries = Object.entries(report.topic_distribution || {}).sort((a, b) => b[1] - a[1]);
  const maxDist = distEntries.length > 0 ? distEntries[0][1] : 1;
  const distBars = distEntries.map(([label, count]) => `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <span style="min-width:100px;font-size:0.84em;text-align:right;color:var(--text)">${sanitizeHtml(label)}</span>
      <div style="flex:1;background:var(--line);border-radius:6px;height:20px;overflow:hidden">
        <div style="background:linear-gradient(90deg,var(--accent),var(--accent-soft));height:100%;border-radius:6px;width:${Math.round((count / maxDist) * 100)}%;transition:width .5s"></div>
      </div>
      <span style="font-size:0.82em;color:var(--muted);min-width:24px">${count}</span>
    </div>`).join("");

  const kwHtml = (report.hot_keywords || []).map((kw) =>
    `<span style="display:inline-block;padding:4px 14px;border-radius:999px;font-size:0.8em;background:var(--surface);border:1px solid var(--line);color:var(--text);margin:3px">${sanitizeHtml(kw)}</span>`
  ).join("");

  const totalPapers = (report.top_picks || []).length + (report.other_papers || []).length;

  return `<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>薩提爾模式每日研究文獻報告 ${dateDisplay}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f6f1e8;--surface:#fffaf2;--line:#d8c5ab;--text:#2b2118;--muted:#766453;--accent:#8c4f2b;--accent-soft:#ead2bf;--card-bg:color-mix(in srgb,var(--surface) 92%,white)}
*{margin:0;padding:0;box-sizing:border-box}
body{background:radial-gradient(circle at top,#fff6ea 0,var(--bg) 55%,#ead8c6 100%);font-family:"Noto Sans TC","PingFang TC","Helvetica Neue",Arial,sans-serif;color:var(--text);min-height:100vh;line-height:1.6}
.container{max-width:880px;margin:0 auto;padding:32px 20px 48px}
h1{font-size:1.6em;text-align:center;margin-bottom:4px;color:var(--text)}
.date-line{text-align:center;color:var(--muted);font-size:0.92em;margin-bottom:28px}
.summary-box{background:var(--card-bg);border-radius:20px;padding:24px 28px;margin-bottom:28px;box-shadow:0 2px 10px rgba(139,79,43,0.06)}
.summary-box h2{font-size:1.1em;color:var(--accent);margin-bottom:10px}
.summary-zh{font-size:0.92em;line-height:1.75;margin-bottom:12px}
.summary-en{font-size:0.84em;color:var(--muted);line-height:1.6}
.trends-box{background:var(--card-bg);border-radius:16px;padding:20px 24px;margin-bottom:28px;box-shadow:0 2px 10px rgba(139,79,43,0.06)}
.trends-box h2{font-size:1.05em;color:var(--accent);margin-bottom:12px}
.trend-item{display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;font-size:0.9em;line-height:1.6}
.trend-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);flex-shrink:0;margin-top:7px}
.section-title{font-size:1.15em;color:var(--accent);margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid var(--accent-soft)}
.footer{text-align:center;padding:32px 0 16px;border-top:1px solid var(--line);margin-top:36px}
.footer a{color:var(--accent);text-decoration:none;font-size:0.88em;margin:0 12px}
.footer a:hover{text-decoration:underline}
.footer-copy{color:var(--muted);font-size:0.78em;margin-top:12px}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.fade-up{animation:fadeUp .5s ease both}
@media(max-width:600px){.container{padding:20px 12px 36px}h1{font-size:1.3em}}
</style>
</head>
<body>
<div class="container fade-up">
<h1>🏔️ 薩提爾模式每日研究文獻報告</h1>
<p class="date-line">${dateDisplay} · 共 ${totalPapers} 篇新文獻</p>

${report.summary ? `<div class="summary-box">
<h2>📋 今日摘要</h2>
<p class="summary-zh">${sanitizeHtml(report.summary.zh || "")}</p>
${report.summary.en ? `<p class="summary-en">${sanitizeHtml(report.summary.en)}</p>` : ""}
</div>` : ""}

${(report.trends || []).length > 0 ? `<div class="trends-box">
<h2>📈 文獻趨勢</h2>
${report.trends.map((t) => `<div class="trend-item"><span class="trend-dot"></span><span>${sanitizeHtml(t)}</span></div>`).join("")}
</div>` : ""}

${(report.top_picks || []).length > 0 ? `<h2 class="section-title">⭐ 重點推薦（TOP Picks）</h2>${topCards}` : ""}

${(report.other_papers || []).length > 0 ? `<h2 class="section-title" style="margin-top:28px">📚 其他文獻</h2>${otherCards}` : ""}

${distEntries.length > 0 ? `<h2 class="section-title" style="margin-top:28px">📊 主題分布</h2>${distBars}` : ""}

${kwHtml ? `<h2 class="section-title" style="margin-top:28px">🏷️ 熱門關鍵詞</h2><div style="text-align:center;margin-bottom:20px">${kwHtml}</div>` : ""}

<div class="footer">
<p>
<a href="https://www.leepsyclinic.com/" target="_blank" rel="noopener noreferrer">🏥 李政洋身心診所首頁</a>
<a href="https://blog.leepsyclinic.com/" target="_blank" rel="noopener noreferrer">📬 訂閱電子報</a>
<a href="https://buymeacoffee.com/CYlee" target="_blank" rel="noopener noreferrer">☕ Buy me a coffee</a>
</p>
<p class="footer-copy">Satir Model Daily Research Report · Powered by GitHub Actions + Zhipu AI</p>
</div>
</div>
</body>
</html>`;
}

async function main() {
  const targetDate = process.env.TARGET_DATE || new Date().toISOString().slice(0, 10);
  const apiKey = process.env.ZHIPU_API_KEY;

  if (!existsSync("papers.json")) {
    console.error("papers.json not found");
    process.exit(1);
  }
  const data = JSON.parse(readFileSync("papers.json", "utf-8"));
  const papers = data.papers || [];
  if (papers.length === 0) {
    console.log("No new papers to report");
    return;
  }

  console.log(`Generating report for ${papers.length} papers...`);

  let report = null;
  if (apiKey) {
    const prompt = buildPrompt(papers);
    report = await callZhipuAI(prompt, 0);
  } else {
    console.log("No API key, generating fallback report");
  }

  if (!report) {
    report = generateFallbackReport(papers, targetDate);
  }

  if (!report.top_picks) report.top_picks = [];
  if (!report.other_papers) report.other_papers = [];

  const html = generateHtml(report, targetDate);

  if (!existsSync("docs")) mkdirSync("docs", { recursive: true });
  writeFileSync(join("docs", `satir-${targetDate}.html`), html);
  console.log(`Report saved: docs/satir-${targetDate}.html`);

  const allPmids = papers.map((p) => p.pmid);
  const pmidFile = join("docs", "processed_pmids.json");
  let existing = [];
  if (existsSync(pmidFile)) {
    try { existing = JSON.parse(readFileSync(pmidFile, "utf-8")).pmids || []; } catch {}
  }
  const merged = [...new Set([...existing, ...allPmids])];
  writeFileSync(pmidFile, JSON.stringify({ pmids: merged, last_updated: targetDate }, null, 2));
  console.log(`Updated processed_pmids.json (${merged.length} total)`);
}

main().catch((err) => {
  console.error("generate_report failed:", err);
  process.exit(1);
});
