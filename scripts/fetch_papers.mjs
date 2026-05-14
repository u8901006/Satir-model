import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
const ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi";

const SEARCH_TERM = [
  '"Satir Model"[Title/Abstract]',
  '"Satir model"[Title/Abstract]',
  '"Satir Growth Model"[Title/Abstract]',
  '"Satir family therapy"[Title/Abstract]',
  '"Satir therapy"[Title/Abstract]',
  '"Satir Transformational Systemic Therapy"[Title/Abstract]',
  '"Human Validation Process Model"[Title/Abstract]',
  '"Satir-based"[Title/Abstract]',
  '"Satir approach"[Title/Abstract]',
  '"Satir group counseling"[Title/Abstract]',
  '"Satir method"[Title/Abstract]',
  '"communication stances"[Title/Abstract] AND Satir[Title/Abstract]',
  '"coping stances"[Title/Abstract] AND Satir[Title/Abstract]',
  '"iceberg model"[Title/Abstract] AND Satir[Title/Abstract]',
  '"family reconstruction"[Title/Abstract] AND Satir[Title/Abstract]',
  '"family sculpting"[Title/Abstract] AND Satir[Title/Abstract]',
  '"parts party"[Title/Abstract] AND Satir[Title/Abstract]',
  "congruence[Title/Abstract] AND Satir[Title/Abstract]",
  '"self-esteem"[Title/Abstract] AND Satir[Title/Abstract]',
].join(" OR ");

function sanitizeString(input) {
  if (typeof input !== "string") return String(input);
  return input.replace(/[<>&'"]/g, (c) => ({
    "<": "&lt;", ">": "&gt;", "&": "&amp;", "'": "&#39;", '"': "&quot;",
  }[c]));
}

function loadProcessedPmids() {
  const filePath = join("docs", "processed_pmids.json");
  if (existsSync(filePath)) {
    try {
      const data = JSON.parse(readFileSync(filePath, "utf-8"));
      return new Set(data.pmids || []);
    } catch {
      return new Set();
    }
  }
  return new Set();
}

function getDateRange() {
  const targetDate = process.env.TARGET_DATE || new Date().toISOString().slice(0, 10);
  const daysBack = parseInt(process.env.DAYS_BACK || "7", 10);
  const endDate = new Date(targetDate);
  const startDate = new Date(endDate);
  startDate.setDate(startDate.getDate() - daysBack);
  const fmt = (d) => `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, "0")}/${String(d.getDate()).padStart(2, "0")}`;
  return { start: fmt(startDate), end: fmt(endDate), target: targetDate };
}

async function fetchJson(url, params) {
  const qs = new URLSearchParams(params).toString();
  const fullUrl = `${url}?${qs}`;
  const res = await fetch(fullUrl, {
    headers: { "User-Agent": "SatirModelBot/1.0 (satir-daily-report)" },
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}

async function searchPmids(dateRange) {
  const params = {
    db: "pubmed",
    term: `(${SEARCH_TERM}) AND ("${dateRange.start}"[Date - Publication] : "${dateRange.end}"[Date - Publication])`,
    retmode: "json",
    retmax: String(process.env.MAX_PAPERS || 40),
    sort: "date",
  };
  const data = await fetchJson(ESEARCH_URL, params);
  const idList = data?.esearchresult?.idlist || [];
  return idList.filter((id) => /^\d+$/.test(id));
}

async function fetchPaperDetails(pmids) {
  if (pmids.length === 0) return [];
  const params = {
    db: "pubmed",
    id: pmids.join(","),
    retmode: "json",
  };
  const data = await fetchJson(ESUMMARY_URL, params);
  const result = data?.result || {};
  const uids = result.uids || [];
  return uids.map((uid) => {
    const p = result[uid] || {};
    return {
      pmid: String(uid),
      title: sanitizeString(p.title || ""),
      journal: sanitizeString(p.fulljournalname || p.source || ""),
      date: p.pubdate || "",
      authors: (p.authors || []).map((a) => sanitizeString(a.name || "")),
      url: `https://pubmed.ncbi.nlm.nih.gov/${uid}/`,
      epubdate: p.epubdate || "",
    };
  }).filter((p) => p.title.length > 0);
}

async function main() {
  const dateRange = getDateRange();
  console.log(`Searching PubMed: ${dateRange.start} to ${dateRange.end}`);

  const pmids = await searchPmids(dateRange);
  console.log(`Found ${pmids.length} PMIDs`);

  const processedPmids = loadProcessedPmids();
  const newPmids = pmids.filter((id) => !processedPmids.has(id));
  console.log(`New (unprocessed) PMIDs: ${newPmids.length}`);

  let papers = [];
  if (newPmids.length > 0) {
    papers = await fetchPaperDetails(newPmids);
  }

  const output = {
    date: dateRange.target,
    date_range: { start: dateRange.start, end: dateRange.end },
    total_found: pmids.length,
    already_processed: pmids.length - newPmids.length,
    count: papers.length,
    papers,
  };

  writeFileSync("papers.json", JSON.stringify(output, null, 2));
  console.log(`Saved ${papers.length} new papers to papers.json`);
}

main().catch((err) => {
  console.error("fetch_papers failed:", err.message);
  writeFileSync("papers.json", JSON.stringify({ date: "error", count: 0, papers: [] }));
  process.exit(0);
});
