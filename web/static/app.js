const $ = (id) => document.getElementById(id);

let config = null;
let activeJobs = new Set();
let browserState = {
  target: null,
  mode: "file",
  current: "",
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `Request failed: ${res.status}`);
  return body;
}

function teacherInputs() {
  return [$("teacher0"), $("teacher1"), $("teacher2")];
}

function fillForm(data) {
  config = data.config;
  $("student").value = config.models.student || "";
  teacherInputs().forEach((input, index) => {
    input.value = config.models.teachers[index] || "";
  });
  const generation = config.generation || {};
  const hardware = config.hardware || {};
  $("bootstrap").value = generation.bootstrap_teacher_index ?? "";
  $("teacher-order").value = (generation.teacher_attempt_order || [0, 1, 2]).join(",");
  $("task-prompt").value = generation.task_system_prompt || "";
  $("system-ram-gb").value = hardware.system_ram_gb ?? 128;
  $("teacher-cache-limit").value = hardware.teacher_cache_limit_gb ?? 45;
  $("bootstrap-max-model").value = hardware.bootstrap_max_model_gb ?? 45;
  $("cache-teachers").checked = hardware.cache_teachers !== false;
  if (data.latest_adapter) $("train-resume").placeholder = data.latest_adapter;
  setSafeOutputDefaults();
}

function timestampSlug() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}`;
}

function setSafeOutputDefaults() {
  const teacherLabel = ["auto", "teacher1", "teacher2", "teacher3"][$("bootstrap").value ? Number($("bootstrap").value) + 1 : 0] || "auto";
  const stamp = timestampSlug();
  if (!$("gen-output").value || $("gen-output").value === "data/generated/gui_claude_tool_traces.jsonl") {
    $("gen-output").value = `data/generated/gui_claude_tool_traces_${teacherLabel}_20_samples_${stamp}.jsonl`;
  }
  if (!$("train-output").value || $("train-output").value === "models/mlx_self_training/gui_train") {
    $("train-output").value = `models/mlx_self_training/gui_train_${stamp}`;
  }
}

function collectConfig() {
  const teachers = teacherInputs().map((input) => input.value.trim()).filter(Boolean);
  const bootstrapValue = $("bootstrap").value;
  const order = $("teacher-order").value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number.parseInt(part, 10));

  return {
    ...config,
    models: {
      student: $("student").value.trim(),
      teachers,
    },
    generation: {
      ...(config.generation || {}),
      bootstrap_teacher_index: bootstrapValue === "" ? null : Number.parseInt(bootstrapValue, 10),
      teacher_attempt_order: order.length ? order : teachers.map((_, index) => index),
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

async function saveConfig() {
  config = collectConfig();
  await api("/api/config", {
    method: "POST",
    body: JSON.stringify({ config }),
  });
  alert("Settings saved.");
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
      train_iters: Number.parseInt($("train-iters").value, 10) || 200,
      train_output: $("train-output").value.trim(),
      resume: $("train-resume").value.trim(),
      overwrite: $("train-overwrite").checked,
    };
  }
  if (type === "report") {
    return {
      type,
      files: $("report-files").value.split(/\s+/).map((part) => part.trim()).filter(Boolean),
    };
  }
  return { type };
}

async function startJob(type) {
  await saveConfig();
  const { job } = await api("/api/jobs", {
    method: "POST",
    body: JSON.stringify(jobPayload(type)),
  });
  activeJobs.add(job.id);
  await refreshJobs();
}

async function stopJob(id) {
  await api(`/api/jobs/${id}/stop`, { method: "POST", body: "{}" });
  await refreshJobs();
}

function setupPathBrowsers() {
  document.querySelectorAll("[data-browse]").forEach((input) => {
    if (input.dataset.browserReady) return;
    input.dataset.browserReady = "1";
    const wrapper = document.createElement("div");
    wrapper.className = "path-row";
    input.parentNode.insertBefore(wrapper, input);
    wrapper.appendChild(input);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "browse-btn";
    button.textContent = "Browse";
    button.addEventListener("click", () => openBrowser(input, input.dataset.browse || "file"));
    wrapper.appendChild(button);
  });

  $("browser-close").addEventListener("click", closeBrowser);
  $("browser-parent").addEventListener("click", () => loadBrowser(browserState.parent || browserState.current));
  $("browser-use-current").addEventListener("click", () => {
    if (!browserState.target) return;
    if (browserState.mode === "save-file") {
      setTargetPath(joinPath(browserState.current, fileNameFromInput(browserState.target.value)));
    } else {
      setTargetPath(browserState.current);
    }
    closeBrowser();
  });
}

function fileNameFromInput(value) {
  const trimmed = String(value || "").trim();
  const parts = trimmed.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) || "output.jsonl";
}

function joinPath(dir, name) {
  return `${String(dir).replace(/[\\/]$/, "")}/${name}`;
}

function setTargetPath(path) {
  if (!browserState.target) return;
  if (browserState.mode === "multi-file") {
    const existing = browserState.target.value.trim();
    browserState.target.value = existing ? `${existing} ${path}` : path;
  } else {
    browserState.target.value = path;
  }
}

function closeBrowser() {
  $("browser-modal").classList.add("hidden");
  browserState.target = null;
}

async function openBrowser(input, mode) {
  browserState = {
    target: input,
    mode,
    current: input.value.trim(),
    parent: "",
  };
  $("browser-use-current").style.display = ["dir", "save-file"].includes(mode) ? "inline-block" : "none";
  $("browser-use-current").textContent = mode === "save-file" ? "Use Directory + Filename" : "Use Current Directory";
  $("browser-modal").classList.remove("hidden");
  await loadBrowser(input.value.trim());
}

async function loadBrowser(path = "") {
  const data = await api(`/api/browse?path=${encodeURIComponent(path)}`);
  browserState.current = data.current;
  browserState.parent = data.parent;
  $("browser-current").textContent = data.current;
  $("browser-parent").disabled = !data.parent;

  $("browser-roots").innerHTML = (data.roots || []).map((root) =>
    `<button type="button" data-root="${escapeHtml(root.path)}">${escapeHtml(root.label)}</button>`
  ).join("");
  document.querySelectorAll("[data-root]").forEach((button) => {
    button.addEventListener("click", () => loadBrowser(button.dataset.root));
  });

  if (data.error) {
    $("browser-list").innerHTML = `<p class="muted">${escapeHtml(data.error)}</p>`;
    return;
  }

  $("browser-list").innerHTML = (data.entries || []).map((entry) => {
    const icon = entry.is_dir ? "[D]" : "[F]";
    const action = entry.is_dir ? "Open" : "Select";
    const disabled = !entry.is_dir && browserState.mode === "dir" ? "disabled" : "";
    return `
      <div class="browser-entry">
        <div>${icon}</div>
        <div>
          <div class="browser-name">${escapeHtml(entry.name)}</div>
          <div class="browser-meta">${escapeHtml(entry.path)}</div>
        </div>
        <button type="button" data-path="${escapeHtml(entry.path)}" data-dir="${entry.is_dir ? "1" : "0"}" ${disabled}>${action}</button>
      </div>
    `;
  }).join("") || "<p class='muted'>No entries.</p>";

  document.querySelectorAll("[data-path]").forEach((button) => {
    button.addEventListener("click", async () => {
      const path = button.dataset.path;
      const isDir = button.dataset.dir === "1";
      if (isDir) {
        await loadBrowser(path);
        return;
      }
      setTargetPath(path);
      closeBrowser();
    });
  });
}

function renderJob(job) {
  const command = (job.command || []).join(" ");
  const log = (job.log || []).join("\n");
  const canStop = ["running", "starting"].includes(job.status);
  return `
    <div class="job">
      <div class="job-head">
        <div>
          <strong>${job.type || "job"} / ${job.id}</strong>
          <div class="status">${job.status}${job.exit_code !== null && job.exit_code !== undefined ? ` (${job.exit_code})` : ""}</div>
        </div>
        ${canStop ? `<button class="danger" data-stop="${job.id}">Stop</button>` : ""}
      </div>
      <p><code>${command}</code></p>
      <pre>${escapeHtml(log)}</pre>
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
  $("jobs").innerHTML = jobs.slice().reverse().map(renderJob).join("") || "<p class='muted'>No jobs yet.</p>";
  document.querySelectorAll("[data-stop]").forEach((button) => {
    button.addEventListener("click", () => stopJob(button.dataset.stop));
  });
  activeJobs = new Set(jobs.filter((job) => ["running", "starting", "stopping"].includes(job.status)).map((job) => job.id));
}

async function init() {
  fillForm(await api("/api/config"));
  setupPathBrowsers();
  $("save-config").addEventListener("click", saveConfig);
  document.querySelectorAll("[data-job]").forEach((button) => {
    button.addEventListener("click", () => startJob(button.dataset.job).catch((error) => alert(error.message)));
  });
  await refreshJobs();
  setInterval(() => {
    if (activeJobs.size) refreshJobs().catch(console.error);
  }, 2000);
}

init().catch((error) => {
  document.body.innerHTML = `<main><pre>${escapeHtml(error.stack || error.message)}</pre></main>`;
});
