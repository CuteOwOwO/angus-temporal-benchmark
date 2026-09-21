const state = {
  mode: "composition",
  compositions: [],
  qa: [],
  annotations: { composition: {}, qa: {} },
  index: 0,
  pendingOnly: false,
  saveTimers: { composition: null, qa: null },
};

const $ = (id) => document.getElementById(id);
const itemId = (item) => state.mode === "composition" ? item.sample_id : item.qa_id;
const allItems = () => state.mode === "composition" ? state.compositions : state.qa;
const currentNotes = () => state.annotations[state.mode];
const visibleItems = () => state.pendingOnly
  ? allItems().filter((item) => !currentNotes()[itemId(item)]?.verdict)
  : allItems();
const current = () => visibleItems()[state.index];
const label = (value) => String(value || "").replaceAll("_", " ");

async function persist(mode = state.mode) {
  clearTimeout(state.saveTimers[mode]);
  $("saveState").textContent = "儲存中";
  const response = await fetch(`/api/annotations/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.annotations[mode]),
  });
  $("saveState").textContent = response.ok ? "已同步" : "儲存失敗";
}

function update(patch, immediate = true) {
  const mode = state.mode;
  const item = current();
  if (!item) return;
  const id = itemId(item);
  currentNotes()[id] = {
    ...(currentNotes()[id] || {}),
    ...patch,
    updated_at: new Date().toISOString(),
  };
  clearTimeout(state.saveTimers[mode]);
  if (immediate) persist(mode);
  else state.saveTimers[mode] = setTimeout(() => persist(mode), 400);
  updateProgress();
  if (state.pendingOnly && patch.verdict) render();
}

function updateProgress() {
  const reviewed = allItems().filter((item) => currentNotes()[itemId(item)]?.verdict).length;
  $("progress").textContent = `${reviewed} / ${allItems().length} 已審`;
}

function renderComposition(item) {
  $("compositionView").hidden = false;
  $("qaView").hidden = true;
  $("itemTitle").textContent = `${item.duration_sec.toFixed(1)} 秒`;
  $("badge").textContent = label(item.source_release_private);
  $("timeline").hidden = true;
  $("revealTimeline").textContent = "顯示事件順序";
  $("eventOrder").innerHTML = item.intervals.map((event) =>
    `<li>${label(event.category)}</li>`
  ).join("");
  $("eventTiming").textContent = item.intervals.map((event) =>
    `${label(event.category)} ${event.start_sec.toFixed(2)}-${event.end_sec.toFixed(2)}s`
  ).join("  /  ");
  $("verdictLegend").textContent = "這段音檔可作為 benchmark 嗎？";
}

function renderQa(item, saved) {
  $("compositionView").hidden = true;
  $("qaView").hidden = false;
  $("itemTitle").textContent = item.question;
  $("badge").textContent = label(item.question_family_private);
  $("question").textContent = item.question;
  $("answers").innerHTML = `<legend>你的答案</legend>${item.options.map((option) =>
    `<label><input type="radio" name="answer" value="${option.key}" ${saved.choice === option.key ? "checked" : ""}><span>${option.key}. ${option.text}</span></label>`
  ).join("")}`;
  document.querySelectorAll('input[name="answer"]').forEach((input) => {
    input.onchange = () => update({ choice: input.value });
  });
  const correct = item.options.find((option) => option.key === item.answer_key_private);
  $("officialAnswer").textContent = `標準答案：${item.answer_key_private}. ${correct?.text || ""}`;
  $("officialAnswer").hidden = true;
  $("revealAnswer").textContent = "顯示標準答案";
  $("verdictLegend").textContent = "這題可作為 benchmark QA 嗎？";
}

function render() {
  const items = visibleItems();
  if (state.index >= items.length) state.index = Math.max(0, items.length - 1);
  const item = current();
  $("review").hidden = !item;
  $("empty").hidden = Boolean(item);
  updateProgress();
  if (!item) return;

  const saved = currentNotes()[itemId(item)] || {};
  $("position").textContent = `${state.index + 1} / ${items.length}  ·  ${itemId(item)}`;
  $("audio").src = `/audio/${item.audio_file.split("/").pop()}`;
  $("note").value = saved.note || "";
  document.querySelectorAll('input[name="verdict"]').forEach((input) => {
    input.checked = saved.verdict === input.value;
  });

  if (state.mode === "composition") renderComposition(item);
  else renderQa(item, saved);
  $("previous").disabled = state.index === 0;
  $("next").disabled = state.index === items.length - 1;
}

function setMode(mode) {
  state.mode = mode;
  state.index = 0;
  $("compositionTab").classList.toggle("active", mode === "composition");
  $("qaTab").classList.toggle("active", mode === "qa");
  $("compositionTab").setAttribute("aria-selected", mode === "composition");
  $("qaTab").setAttribute("aria-selected", mode === "qa");
  render();
}

$("compositionTab").onclick = () => setMode("composition");
$("qaTab").onclick = () => setMode("qa");
$("pendingOnly").onchange = (event) => {
  state.pendingOnly = event.target.checked;
  state.index = 0;
  render();
};
$("previous").onclick = () => { state.index = Math.max(0, state.index - 1); render(); };
$("next").onclick = () => { state.index = Math.min(visibleItems().length - 1, state.index + 1); render(); };
$("revealTimeline").onclick = () => {
  $("timeline").hidden = !$("timeline").hidden;
  $("revealTimeline").textContent = $("timeline").hidden ? "顯示事件順序" : "隱藏事件順序";
};
$("revealAnswer").onclick = () => {
  $("officialAnswer").hidden = !$("officialAnswer").hidden;
  $("revealAnswer").textContent = $("officialAnswer").hidden ? "顯示標準答案" : "隱藏標準答案";
};
document.querySelectorAll('input[name="verdict"]').forEach((input) => {
  input.onchange = () => update({ verdict: input.value });
});
$("note").oninput = () => update({ note: $("note").value.trim() }, false);

Promise.all([
  fetch("/api/compositions").then((response) => response.json()),
  fetch("/api/qa").then((response) => response.json()),
  fetch("/api/annotations/composition").then((response) => response.json()),
  fetch("/api/annotations/qa").then((response) => response.json()),
]).then(([compositions, qa, compositionNotes, qaNotes]) => {
  state.compositions = compositions;
  state.qa = qa;
  state.annotations.composition = compositionNotes;
  state.annotations.qa = qaNotes;
  render();
}).catch((error) => {
  $("progress").textContent = "載入失敗";
  $("empty").hidden = false;
  $("empty").querySelector("p").textContent = error.message;
});
