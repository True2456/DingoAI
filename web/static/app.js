const $ = (id) => document.getElementById(id);

let config = null;
let presets = { presets: {} };
let activeJobs = new Set();
/** @type {Map<string, { pinned: boolean }>} */
const logScrollState = new Map();

/** @type {Map<string, { hidden: HTMLInputElement, mode: string, dirEl: HTMLInputElement, nameEl?: HTMLInputElement, multiEl?: HTMLTextAreaElement }>} */
const pathFields = new Map();

let browserState = { target: null, field: null, mode: "file", current: "", parent: "" };
let pendingJobType = null;
let pendingPresetSave = null;

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const contentType = res.headers.get("Content-Type") || "";
  if (contentType.includes("application/json")) {
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || `Request failed: ${res.status}`);
    return body;
  }
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res;
}

function toast(message, type = "info") {
  const stack = $("toast-stack");
  const el = document.createElement("div");
  el.className = `toast ${type === "error" ? "error" : type === "success" ? "success" : ""}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

function teacherInputs() {
  return [$("teacher0"), $("teacher1"), $("teacher2")];
}

function displayTeacherOrder(order, teacherCount) {
  const values = order && order.length ? order : Array.from({ length: teacherCount }, (_, i) => i + 1);
  const isZeroBased = values.some((v) => v === 0);
  return values.map((v) => (isZeroBased ? v + 1 : v));
}

function timestampSlug() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}`;
}

function setSafeOutputDefaults() {
  const teacherLabel =
    ["auto", "teacher1", "teacher2", "teacher3"][$("bootstrap").value ? Number($("bootstrap").value) + 1 : 0] ||
    "auto";
  const stamp = timestampSlug();
  const genDefault = "data/generated/dingo_run.jsonl";
  const trainDefault = "models/mlx_self_training/dingo_train";
  if (!$("gen-output").value || $("gen-output").value.includes("gui_")) {
    setPathValue("gen-output", `data/generated/dingo_${teacherLabel}_${stamp}.jsonl`);
  }
  if (!$("train-output").value || $("train-output").value.includes("gui_")) {
    setPathValue("train-output", `models/mlx_self_training/dingo_train_${stamp}`);
  }
  if (!$("gen-output").value) setPathValue("gen-output", genDefault);
  if (!$("train-output").value) setPathValue("train-output", trainDefault);
  syncAllPathFields();
}

function fillForm(data) {
  config = data.config;
  setPathValue("student", config.models.student || "");
  teacherInputs().forEach((input, index) => {
    setPathValue(input.id, config.models.teachers[index] || "");
  });
  const generation = config.generation || {};
  const hardware = config.hardware || {};
  $("bootstrap").value = generation.bootstrap_teacher_index ?? "";
  $("teacher-order").value = displayTeacherOrder(generation.teacher_attempt_order, teacherInputs().length).join(",");
  $("task-prompt").value = generation.task_system_prompt || "";
  $("system-ram-gb").value = hardware.system_ram_gb ?? 128;
  $("teacher-cache-limit").value = hardware.teacher_cache_limit_gb ?? 45;
  $("bootstrap-max-model").value = hardware.bootstrap_max_model_gb ?? 45;
  $("cache-teachers").checked = hardware.cache_teachers !== false;
  if (data.latest_adapter) {
    const resume = $("train-resume");
    if (pathFields.has("train-resume")) {
      pathFields.get("train-resume").dirEl.placeholder = data.latest_adapter;
    } else {
      resume.placeholder = data.latest_adapter;
    }
  }
  setSafeOutputDefaults();
  syncAllPathFields();
  updateTrainRatio();
}

function collectConfig() {
  const teachers = teacherInputs().map((i) => i.value.trim()).filter(Boolean);
  const bootstrapValue = $("bootstrap").value;
  const order = $("teacher-order").value
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => Number.parseInt(p, 10));

  return {
    ...config,
    models: { student: $("student").value.trim(), teachers },
    generation: {
      ...(config.generation || {}),
      bootstrap_teacher_index: bootstrapValue === "" ? null : Number.parseInt(bootstrapValue, 10),
      teacher_attempt_order: order.length ? order : teachers.map((_, i) => i + 1),
      task_system_prompt: $("task-prompt").value,
    },
    hardware: {
      ...(config.hardware || {}),
      memory_profile: "custom",
      system_ram_gb: Number.parseFloat($("system-ram-gb").value) || 128,
      teacher_cache_limit_gb: Number.parseFloat($("teacher-cache-limit").value) || 0,
      bootstrap_max_model_gb: Number.parseFloat($("bootstrap-max-model").value) || 1,
      cache_teachers: $("cache-teachers").checked,
    },
  };
}

function collectModelsSlice() {
  const full = collectConfig();
  return {
    models: full.models,
    generation: {
      bootstrap_teacher_index: full.generation.bootstrap_teacher_index,
      teacher_attempt_order: full.generation.teacher_attempt_order,
    },
    hardware: full.hardware,
  };
}

async function saveConfig() {
  commitAllPathFields();
  config = collectConfig();
  await api("/api/config", { method: "POST", body: JSON.stringify({ config }) });
  toast("Settings saved to config.", "success");
}

function presetsOfType(type) {
  return Object.entries(presets.presets || {})
    .filter(([, p]) => p.type === type || (type === "prompt" && p.type === "combined") || (type === "models" && p.type === "combined"))
    .sort(([a], [b]) => a.localeCompare(b));
}

function refreshPresetSelects() {
  for (const [selectId, type] of [
    ["models-preset-select", "models"],
    ["prompt-preset-select", "prompt"],
  ]) {
    const select = $(selectId);
    const current = select.value;
    select.innerHTML = '<option value="">— load preset —</option>';
    for (const [name, preset] of presetsOfType(type)) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = preset.label || name;
      select.appendChild(opt);
    }
    if ([...select.options].some((o) => o.value === current)) select.value = current;
  }
}

async function loadPresetsFromServer() {
  presets = await api("/api/presets");
  refreshPresetSelects();
}

function applyModelsPreset(preset) {
  if (preset.models) {
    setPathValue("student", preset.models.student || "");
    teacherInputs().forEach((input, i) => {
      setPathValue(input.id, (preset.models.teachers || [])[i] || "");
    });
  }
  syncAllPathFields();
  const gen = preset.generation || {};
  if (gen.bootstrap_teacher_index !== undefined && gen.bootstrap_teacher_index !== null) {
    $("bootstrap").value = String(gen.bootstrap_teacher_index);
  } else if (gen.bootstrap_teacher_index === null) {
    $("bootstrap").value = "";
  }
  if (gen.teacher_attempt_order) {
    $("teacher-order").value = displayTeacherOrder(gen.teacher_attempt_order, teacherInputs().length).join(",");
  }
  const hw = preset.hardware || {};
  if (hw.system_ram_gb !== undefined) $("system-ram-gb").value = hw.system_ram_gb;
  if (hw.teacher_cache_limit_gb !== undefined) $("teacher-cache-limit").value = hw.teacher_cache_limit_gb;
  if (hw.bootstrap_max_model_gb !== undefined) $("bootstrap-max-model").value = hw.bootstrap_max_model_gb;
  if (hw.cache_teachers !== undefined) $("cache-teachers").checked = hw.cache_teachers;
}

function applyPromptPreset(preset) {
  if (preset.task_system_prompt !== undefined) {
    $("task-prompt").value = preset.task_system_prompt;
  }
}

function loadPresetFromSelect(selectId, type) {
  const name = $(selectId).value;
  if (!name) {
    toast("Choose a preset first.", "error");
    return;
  }
  const preset = presets.presets[name];
  if (!preset) {
    toast("Preset not found.", "error");
    return;
  }
  if (preset.type === "combined") {
    if (type === "models") applyModelsPreset(preset);
    else applyPromptPreset(preset);
  } else if (type === "models") {
    applyModelsPreset(preset);
  } else {
    applyPromptPreset(preset);
  }
  toast(`Loaded preset “${preset.label || name}”.`, "success");
}

function openPresetSaveModal(kind) {
  pendingPresetSave = kind;
  $("preset-modal-title").textContent =
    kind === "models" ? "Save models preset" : kind === "prompt" ? "Save prompt preset" : "Save combined preset";
  $("preset-name").value = "";
  $("preset-label").value = "";
  $("preset-description").value = "";
  $("preset-modal").classList.remove("hidden");
  $("preset-name").focus();
}

async function savePresetFromModal() {
  const kind = pendingPresetSave;
  const name = $("preset-name").value.trim();
  if (!name) {
    toast("Preset name is required.", "error");
    return;
  }
  const payload = {
    name,
    type: kind === "combined" ? "combined" : kind,
    label: $("preset-label").value.trim() || name,
    description: $("preset-description").value.trim(),
  };
  if (kind === "models" || kind === "combined") {
    Object.assign(payload, collectModelsSlice());
    payload.models = collectConfig().models;
  }
  if (kind === "prompt" || kind === "combined") {
    payload.task_system_prompt = $("task-prompt").value;
  }
  await api("/api/presets", { method: "POST", body: JSON.stringify(payload) });
  $("preset-modal").classList.add("hidden");
  pendingPresetSave = null;
  await loadPresetsFromServer();
  const slug = name.toLowerCase().replace(/[^a-z0-9-_]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
  const selectId = kind === "prompt" ? "prompt-preset-select" : "models-preset-select";
  $(selectId).value = slug;
  toast(`Saved preset “${payload.label}”.`, "success");
}

async function deletePreset(selectId) {
  const name = $(selectId).value;
  if (!name) {
    toast("Choose a preset to delete.", "error");
    return;
  }
  const preset = presets.presets[name];
  if (!confirm(`Delete preset “${preset?.label || name}”?`)) return;
  await api(`/api/presets/${encodeURIComponent(name)}`, { method: "DELETE" });
  await loadPresetsFromServer();
  toast("Preset deleted.", "success");
}

async function updateTrainRatio() {
  const el = $("train-ratio");
  const dataPath = $("train-data").value.trim();
  const iters = Number.parseInt($("train-iters").value, 10) || 0;
  el.textContent = "Ratio: enter a JSONL path to estimate";
  el.className = "ratio-hint muted";
  if (!dataPath || !iters) return;
  try {
    const stats = await api(`/api/file-stats?path=${encodeURIComponent(dataPath)}`);
    const samples = stats.lines || 1;
    const ratio = iters / samples;
    el.textContent = `Ratio: ${ratio.toFixed(1)}× (${iters} iters ÷ ${samples} samples)`;
    el.className = "ratio-hint " + (ratio > 3 ? "danger" : ratio > 2 ? "warn" : "safe");
  } catch {
    el.textContent = `Ratio: ${iters} iters (could not read file yet)`;
  }
}

function jobPayload(type) {
  if (type === "generate") {
    return {
      type,
      samples: Number.parseInt($("gen-samples").value, 10) || 20,
      output: $("gen-output").value.trim(),
      overwrite: $("gen-overwrite").checked,
    };
  }
  if (type === "train") {
    return {
      type,
      data: $("train-data").value.trim(),
      train_iters: Number.parseInt($("train-iters").value, 10) || 120,
      train_output: $("train-output").value.trim(),
      resume: $("train-resume").value.trim(),
      overwrite: $("train-overwrite").checked,
    };
  }
  if (type === "report") {
    return {
      type,
      files: $("report-files").value.split(/\s+/).map((p) => p.trim()).filter(Boolean),
    };
  }
  return { type };
}

function describeCommand(payload) {
  if (payload.type === "generate") {
    return `generate-only · ${payload.samples} samples → ${payload.output}${payload.overwrite ? " · overwrite" : ""}`;
  }
  if (payload.type === "train") {
    return `train-only · ${payload.data} · ${payload.train_iters} iters → ${payload.train_output}`;
  }
  if (payload.type === "report") return `report · ${(payload.files || []).join(" ")}`;
  return payload.type;
}

function needsSafetyConfirm(type, payload) {
  if (payload.overwrite) return `Overwrite is enabled. Existing output may be replaced.\n\n`;
  if (type === "generate" && payload.output) return true;
  if (type === "train" && payload.train_output) return true;
  if (["smoke", "full", "resume"].includes(type)) return true;
  return false;
}

function showConfirm(message, commandText) {
  return new Promise((resolve) => {
    $("confirm-message").textContent = message;
    const cmdEl = $("confirm-command");
    if (commandText) {
      cmdEl.textContent = commandText;
      cmdEl.classList.remove("hidden");
    } else {
      cmdEl.classList.add("hidden");
    }
    $("confirm-modal").classList.remove("hidden");
    const onOk = () => {
      cleanup();
      resolve(true);
    };
    const onCancel = () => {
      cleanup();
      resolve(false);
    };
    const cleanup = () => {
      $("confirm-modal").classList.add("hidden");
      $("confirm-ok").removeEventListener("click", onOk);
      $("confirm-cancel").removeEventListener("click", onCancel);
    };
    $("confirm-ok").addEventListener("click", onOk);
    $("confirm-cancel").addEventListener("click", onCancel);
  });
}

async function startJob(type) {
  await saveConfig();
  const payload = jobPayload(type);
  if (type === "train" && !payload.data) {
    toast("Training requires an input JSONL path.", "error");
    return;
  }
  if (type === "report" && !payload.files?.length) {
    toast("Add at least one JSONL for the report.", "error");
    return;
  }

  let confirmMsg = document.querySelector(`[data-job="${type}"]`)?.dataset?.confirm || `Start ${type}?`;
  if (payload.overwrite) {
    confirmMsg = `Overwrite is ON.\n\n${confirmMsg}`;
  }
  if (needsSafetyConfirm(type, payload)) {
    const ok = await showConfirm(confirmMsg, describeCommand(payload));
    if (!ok) return;
  }

  const { job } = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
  logScrollState.set(job.id, { pinned: false });
  activeJobs.add(job.id);
  toast(`Job ${job.id} started (${job.type}).`, "success");
  await refreshJobs();
}

async function stopJob(id) {
  if (!confirm("Stop this job?")) return;
  await api(`/api/jobs/${id}/stop`, { method: "POST", body: "{}" });
  toast("Stop requested.", "success");
  await refreshJobs();
}

async function clearJobLog(id) {
  if (!confirm("Clear this job's log display?")) return;
  await api(`/api/jobs/${id}/clear-log`, { method: "POST", body: "{}" });
  logScrollState.set(id, { pinned: false });
  await refreshJobs();
}

function downloadJobLog(id) {
  window.open(`/api/jobs/${id}/log`, "_blank");
}

function isLogAtBottom(logEl) {
  return logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 12;
}

function attachLogScrollHandlers() {
  document.querySelectorAll(".job-log").forEach((logEl) => {
    if (logEl.dataset.scrollBound) return;
    logEl.dataset.scrollBound = "1";
    const jobId = logEl.closest(".job")?.dataset?.jobId;
    if (!jobId) return;
    logEl.addEventListener(
      "scroll",
      () => {
        const state = logScrollState.get(jobId) || { pinned: false };
        state.pinned = !isLogAtBottom(logEl);
        logScrollState.set(jobId, state);
      },
      { passive: true },
    );
  });
}

function splitPathParts(fullPath, mode) {
  const trimmed = String(fullPath || "").trim().replace(/\\/g, "/");
  if (!trimmed) {
    return { dir: "", name: mode === "save-file" ? "dingo_run.jsonl" : "" };
  }
  if (mode === "dir") return { dir: trimmed, name: "" };
  const slash = trimmed.lastIndexOf("/");
  if (slash === -1) return { dir: "", name: trimmed };
  return { dir: trimmed.slice(0, slash), name: trimmed.slice(slash + 1) };
}

function joinPathParts(dir, name) {
  const d = String(dir || "").trim().replace(/\\/g, "/").replace(/\/$/, "");
  const n = String(name || "").trim();
  if (!d) return n;
  if (!n) return d;
  return `${d}/${n}`;
}

function getPathValue(id) {
  const hidden = $(id);
  return hidden ? hidden.value.trim() : "";
}

function setPathValue(id, fullPath) {
  const hidden = $(id);
  if (!hidden) return;
  hidden.value = fullPath;
  syncPathFieldUI(id);
}

function syncPathFieldUI(id) {
  const field = pathFields.get(id);
  const hidden = $(id);
  if (!field || !hidden) return;
  const { dir, name } = splitPathParts(hidden.value, field.mode);
  field.dirEl.value = dir;
  field.dirEl.placeholder = dir ? "" : "Select a folder…";
  if (field.nameEl) {
    field.nameEl.value = name;
  }
  if (field.multiEl) {
    field.multiEl.value = hidden.value.trim().split(/\s+/).filter(Boolean).join("\n");
  }
}

function syncAllPathFields() {
  pathFields.forEach((_, id) => syncPathFieldUI(id));
}

function commitPathField(id) {
  const field = pathFields.get(id);
  const hidden = $(id);
  if (!field || !hidden) return;
  if (field.mode === "multi-file") return;
  if (field.mode === "dir") {
    hidden.value = field.dirEl.value.trim();
    return;
  }
  hidden.value = joinPathParts(field.dirEl.value, field.nameEl?.value || "");
  if (id === "train-data") updateTrainRatio().catch(() => {});
}

function buildPathField(hiddenInput) {
  const id = hiddenInput.id;
  const mode = hiddenInput.dataset.browse || "file";
  hiddenInput.type = "hidden";

  const wrap = document.createElement("div");
  wrap.className = "path-field";

  const dirRow = document.createElement("div");
  dirRow.className = "path-dir-row";

  const dirInput = document.createElement("input");
  dirInput.type = "text";
  dirInput.className = "path-dir";
  dirInput.readOnly = true;
  dirInput.tabIndex = -1;
  dirInput.placeholder = "Select a folder…";
  dirInput.setAttribute("aria-readonly", "true");

  const browseBtn = document.createElement("button");
  browseBtn.type = "button";
  browseBtn.className = "browse-btn secondary";
  browseBtn.textContent = "Browse";
  browseBtn.addEventListener("click", () => openBrowser(id));

  dirRow.appendChild(dirInput);
  dirRow.appendChild(browseBtn);
  wrap.appendChild(dirRow);

  let nameEl;
  let multiEl;

  if (mode === "file" || mode === "save-file") {
    const nameRow = document.createElement("div");
    nameRow.className = "path-name-row";
    const nameLabel = document.createElement("label");
    nameLabel.textContent = "File name";
    nameEl = document.createElement("input");
    nameEl.type = "text";
    nameEl.className = "path-name";
    nameEl.placeholder = mode === "save-file" ? "dingo_run.jsonl" : "all_tool_training.jsonl";
    nameEl.addEventListener("input", () => commitPathField(id));
    nameLabel.appendChild(nameEl);
    nameRow.appendChild(nameLabel);
    wrap.appendChild(nameRow);
  }

  if (mode === "multi-file") {
    multiEl = document.createElement("textarea");
    multiEl.className = "path-multi-readonly";
    multiEl.readOnly = true;
    multiEl.placeholder = "Use Browse to add JSONL files…";
    wrap.appendChild(multiEl);
    const actions = document.createElement("div");
    actions.className = "path-multi-actions";
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "secondary";
    clearBtn.textContent = "Clear list";
    clearBtn.addEventListener("click", () => {
      hiddenInput.value = "";
      syncPathFieldUI(id);
    });
    actions.appendChild(clearBtn);
    wrap.appendChild(actions);
  }

  hiddenInput.parentNode.insertBefore(wrap, hiddenInput.nextSibling);
  pathFields.set(id, { hidden: hiddenInput, mode, dirEl: dirInput, nameEl, multiEl });
  syncPathFieldUI(id);
}

function setupPathBrowsers() {
  document.querySelectorAll("[data-browse]").forEach((input) => {
    if (input.dataset.pathFieldReady) return;
    input.dataset.pathFieldReady = "1";
    buildPathField(input);
  });

  $("browser-close").addEventListener("click", closeBrowser);
  $("browser-parent").addEventListener("click", () => loadBrowser(browserState.parent || browserState.current));
  $("browser-use-current").addEventListener("click", () => {
    if (!browserState.field) return;
    applyBrowseDir(browserState.current);
    closeBrowser();
  });
}

function applyBrowseDir(dir) {
  const field = browserState.field;
  if (!field) return;
  field.dirEl.value = dir;
  if (field.mode === "multi-file") return;
  commitPathField(field.hidden.id);
}

function commitAllPathFields() {
  pathFields.forEach((_, id) => commitPathField(id));
}

function applyBrowseFile(filePath) {
  const field = browserState.field;
  if (!field) return;
  if (field.mode === "multi-file") {
    const existing = field.hidden.value.trim();
    const paths = existing ? existing.split(/\s+/) : [];
    if (!paths.includes(filePath)) paths.push(filePath);
    field.hidden.value = paths.join(" ");
    syncPathFieldUI(field.hidden.id);
    return;
  }
  const { dir, name } = splitPathParts(filePath, field.mode === "dir" ? "dir" : "file");
  field.dirEl.value = dir;
  if (field.nameEl) field.nameEl.value = name;
  commitPathField(field.hidden.id);
}

function closeBrowser() {
  $("browser-modal").classList.add("hidden");
  browserState.target = null;
  browserState.field = null;
}

async function openBrowser(fieldId) {
  const field = pathFields.get(fieldId);
  if (!field) return;
  commitPathField(fieldId);
  const startDir = field.dirEl.value.trim() || splitPathParts(field.hidden.value, field.mode).dir;
  browserState = {
    target: field.hidden,
    field,
    mode: field.mode,
    current: startDir,
    parent: "",
  };
  const showDirPick = ["dir", "save-file", "file", "multi-file"].includes(field.mode);
  $("browser-use-current").style.display = showDirPick ? "inline-block" : "none";
  $("browser-use-current").textContent = "Select this folder";
  $("browser-modal").classList.remove("hidden");
  await loadBrowser(startDir);
}

async function loadBrowser(path = "") {
  const data = await api(`/api/browse?path=${encodeURIComponent(path)}`);
  browserState.current = data.current;
  browserState.parent = data.parent;
  $("browser-current").textContent = data.current;
  $("browser-parent").disabled = !data.parent;

  $("browser-roots").innerHTML = (data.roots || [])
    .map((root) => `<button type="button" class="secondary" data-root="${escapeHtml(root.path)}">${escapeHtml(root.label)}</button>`)
    .join("");
  document.querySelectorAll("[data-root]").forEach((button) => {
    button.addEventListener("click", () => loadBrowser(button.dataset.root));
  });

  if (data.error) {
    $("browser-list").innerHTML = `<p class="muted">${escapeHtml(data.error)}</p>`;
    return;
  }

  $("browser-list").innerHTML =
    (data.entries || [])
      .map((entry) => {
        const icon = entry.is_dir ? "📁" : "📄";
        const action = entry.is_dir ? "Open" : "Select";
        const disabled = !entry.is_dir && browserState.mode === "dir" ? "disabled" : "";
        return `
      <div class="browser-entry">
        <div>${icon}</div>
        <div>
          <div class="browser-name">${escapeHtml(entry.name)}</div>
          <div class="browser-meta">${escapeHtml(entry.path)}</div>
        </div>
        <button type="button" class="secondary" data-path="${escapeHtml(entry.path)}" data-dir="${entry.is_dir ? "1" : "0"}" ${disabled}>${action}</button>
      </div>`;
      })
      .join("") || "<p class='muted'>No entries.</p>";

  document.querySelectorAll("[data-path]").forEach((button) => {
    button.addEventListener("click", async () => {
      const path = button.dataset.path;
      if (button.dataset.dir === "1") {
        await loadBrowser(path);
        return;
      }
      applyBrowseFile(path);
      if (browserState.mode !== "multi-file") closeBrowser();
    });
  });
}

function statusClass(status) {
  return `status status-${status || "unknown"}`;
}

function renderJob(job) {
  const command = (job.command || []).join(" ");
  const log = (job.log || []).join("\n");
  const canStop = ["running", "starting"].includes(job.status);
  return `
    <div class="job" data-job-id="${escapeHtml(job.id)}">
      <div class="job-head">
        <div>
          <strong>${escapeHtml(job.type || "job")} / ${escapeHtml(job.id)}</strong>
          <div class="${statusClass(job.status)}">${escapeHtml(job.status)}${
    job.exit_code !== null && job.exit_code !== undefined ? ` (${job.exit_code})` : ""
  }</div>
        </div>
        <div class="job-actions">
          <button type="button" class="secondary" data-download="${job.id}">Download log</button>
          <button type="button" class="secondary" data-clear="${job.id}">Clear log</button>
          ${canStop ? `<button type="button" class="danger" data-stop="${job.id}">Stop</button>` : ""}
        </div>
      </div>
      <p><code>${escapeHtml(command)}</code></p>
      <div class="job-log-wrap">
        <pre class="job-log" tabindex="0" aria-label="Job log">${escapeHtml(log)}</pre>
      </div>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function refreshJobs() {
  const { jobs } = await api("/api/jobs");
  const scrollTops = new Map();
  document.querySelectorAll(".job-log").forEach((logEl) => {
    const jobId = logEl.closest(".job")?.dataset?.jobId;
    if (!jobId) return;
    scrollTops.set(jobId, {
      scrollTop: logEl.scrollTop,
      atBottom: isLogAtBottom(logEl),
    });
  });

  $("jobs").innerHTML = jobs.slice().reverse().map(renderJob).join("") || "<p class='muted'>No jobs yet.</p>";

  document.querySelectorAll("[data-stop]").forEach((btn) => {
    btn.addEventListener("click", () => stopJob(btn.dataset.stop));
  });
  document.querySelectorAll("[data-download]").forEach((btn) => {
    btn.addEventListener("click", () => downloadJobLog(btn.dataset.download));
  });
  document.querySelectorAll("[data-clear]").forEach((btn) => {
    btn.addEventListener("click", () => clearJobLog(btn.dataset.clear));
  });

  activeJobs = new Set(jobs.filter((j) => ["running", "starting", "stopping"].includes(j.status)).map((j) => j.id));

  document.querySelectorAll(".job-log").forEach((logEl) => {
    const jobId = logEl.closest(".job")?.dataset?.jobId;
    if (!jobId) return;
    const prev = scrollTops.get(jobId);
    const state = logScrollState.get(jobId) || { pinned: false };
    logScrollState.set(jobId, state);

    if (!state.pinned) {
      logEl.scrollTop = logEl.scrollHeight;
    } else if (prev) {
      logEl.scrollTop = prev.scrollTop;
    }
  });

  attachLogScrollHandlers();
}

async function init() {
  setupPathBrowsers();
  fillForm(await api("/api/config"));
  await loadPresetsFromServer().catch(() => {
    presets = { presets: {} };
    refreshPresetSelects();
  });

  $("save-config").addEventListener("click", () => saveConfig().catch((e) => toast(e.message, "error")));

  $("models-preset-load").addEventListener("click", () => loadPresetFromSelect("models-preset-select", "models"));
  $("prompt-preset-load").addEventListener("click", () => loadPresetFromSelect("prompt-preset-select", "prompt"));
  $("models-preset-save").addEventListener("click", () => openPresetSaveModal("models"));
  $("prompt-preset-save").addEventListener("click", () => openPresetSaveModal("prompt"));
  $("models-preset-delete").addEventListener("click", () => deletePreset("models-preset-select").catch((e) => toast(e.message, "error")));
  $("prompt-preset-delete").addEventListener("click", () => deletePreset("prompt-preset-select").catch((e) => toast(e.message, "error")));

  $("preset-cancel").addEventListener("click", () => {
    $("preset-modal").classList.add("hidden");
    pendingPresetSave = null;
  });
  $("preset-ok").addEventListener("click", () => savePresetFromModal().catch((e) => toast(e.message, "error")));

  document.querySelectorAll("[data-job]").forEach((button) => {
    button.addEventListener("click", () => startJob(button.dataset.job).catch((e) => toast(e.message, "error")));
  });

  $("train-iters").addEventListener("input", () => updateTrainRatio().catch(() => {}));
  $("train-data").addEventListener("input", () => updateTrainRatio().catch(() => {}));
  updateTrainRatio().catch(() => {});

  $("confirm-cancel").addEventListener("click", () => $("confirm-modal").classList.add("hidden"));

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeBrowser();
      $("confirm-modal").classList.add("hidden");
      $("preset-modal").classList.add("hidden");
    }
  });

  await refreshJobs();
  setInterval(() => {
    if (activeJobs.size) refreshJobs().catch(console.error);
  }, 2000);
}

init().catch((error) => {
  document.body.innerHTML = `<main class="card"><pre>${escapeHtml(error.stack || error.message)}</pre></main>`;
});
