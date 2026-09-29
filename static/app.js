(() => {
  "use strict";

  // ---------------------------------------------------------------- state

  const state = {
    history: [], // [{role, text}]
    lastQuestion: null,
  };

  // ---------------------------------------------------------------- DOM

  const el = {
    fileInput: document.getElementById("file-input"),
    uploadZone: document.getElementById("upload-zone"),
    uploadInner: document.getElementById("upload-inner"),
    docList: document.getElementById("doc-list"),
    docListEmpty: document.getElementById("doc-list-empty"),
    statDocs: document.getElementById("stat-documents"),
    statChunks: document.getElementById("stat-chunks"),
    statusDotQdrant: document.getElementById("status-dot-qdrant"),
    statusTextQdrant: document.getElementById("status-text-qdrant"),
    statusDotGemini: document.getElementById("status-dot-gemini"),
    statusTextGemini: document.getElementById("status-text-gemini"),
    messages: document.getElementById("messages"),
    emptyState: document.getElementById("empty-state"),
    composer: document.getElementById("composer"),
    questionInput: document.getElementById("question-input"),
    sendBtn: document.getElementById("send-btn"),
    clearChatBtn: document.getElementById("clear-chat-btn"),
    tplMessage: document.getElementById("tpl-message"),
    tplSource: document.getElementById("tpl-source"),
    tplDoc: document.getElementById("tpl-doc"),
    mobileTabs: document.querySelectorAll(".mobile-tab"),
    panels: document.querySelectorAll("[data-panel]"),
  };

  // ---------------------------------------------------------------- utils

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // Minimal, safe markdown rendering: bold, italics, inline code, code
  // blocks, links stripped of scheme risk, and paragraph/line breaks.
  function renderMarkdown(text) {
    let escaped = escapeHtml(text);

    escaped = escaped.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
    escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    escaped = escaped.replace(/(^|\s)\*([^*]+)\*(?=\s|$)/g, "$1<em>$2</em>");

    // Bullet lists
    const lines = escaped.split("\n");
    let html = "";
    let inList = false;
    for (const line of lines) {
      const bulletMatch = line.match(/^\s*[-*]\s+(.*)/);
      if (bulletMatch) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${bulletMatch[1]}</li>`;
      } else {
        if (inList) { html += "</ul>"; inList = false; }
        if (line.trim() === "") {
          html += "";
        } else {
          html += `<p>${line}</p>`;
        }
      }
    }
    if (inList) html += "</ul>";
    return html || `<p></p>`;
  }

  function formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function formatDate(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString([], { month: "short", day: "numeric" });
    } catch {
      return "";
    }
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    let body;
    try {
      body = await res.json();
    } catch {
      body = { success: false, error: { code: "BAD_RESPONSE", message: "Unexpected server response." } };
    }
    if (!res.ok || !body.success) {
      const err = body.error || { code: "UNKNOWN", message: "Something went wrong." };
      throw new Error(err.message || "Something went wrong.");
    }
    return body.data;
  }

  // ---------------------------------------------------------------- mobile tabs

  el.mobileTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      el.mobileTabs.forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      const target = tab.dataset.panel;
      el.panels.forEach((p) => p.classList.toggle("is-visible", p.dataset.panel === target));
    });
  });
  // default visible panel on mobile
  document.querySelector('[data-panel="library"]').classList.add("is-visible");

  // ---------------------------------------------------------------- health

  async function refreshHealth() {
    try {
      const data = await api("/api/health");
      setStatus(el.statusDotQdrant, el.statusTextQdrant, data.qdrant.connected, "Connected", "Unavailable");
      setStatus(el.statusDotGemini, el.statusTextGemini, data.gemini.configured, "Configured", "Not configured");
      el.statDocs.textContent = data.knowledge_base.documents;
      el.statChunks.textContent = data.knowledge_base.chunks;
    } catch {
      setStatus(el.statusDotQdrant, el.statusTextQdrant, false, "Connected", "Unavailable");
      setStatus(el.statusDotGemini, el.statusTextGemini, false, "Configured", "Not configured");
    }
  }

  function setStatus(dotEl, textEl, ok, okText, badText) {
    dotEl.classList.remove("ok", "error");
    dotEl.classList.add(ok ? "ok" : "error");
    textEl.textContent = ok ? okText : badText;
  }

  // ---------------------------------------------------------------- documents

  function docIconClass(fileType) {
    if (fileType === "pdf") return "pdf";
    if (fileType === "docx") return "docx";
    return "txt";
  }

  function statusLabel(doc) {
    const map = {
      uploading: "Uploading…",
      extracting: "Extracting text…",
      chunking: "Splitting into passages…",
      embedding: "Generating embeddings…",
      indexing: "Indexing…",
      completed: `${doc.chunk_count} passage${doc.chunk_count === 1 ? "" : "s"}`,
      failed: doc.error_message ? `Failed — ${doc.error_message}` : "Failed",
    };
    return map[doc.status] || doc.status;
  }

  function statusClass(status) {
    if (status === "completed") return "status-completed";
    if (status === "failed") return "status-failed";
    return "status-working";
  }

  function renderDocuments(docs) {
    el.docList.querySelectorAll(".doc-row").forEach((n) => n.remove());
    el.docListEmpty.style.display = docs.length ? "none" : "block";

    for (const doc of docs) {
      const node = el.tplDoc.content.cloneNode(true);
      const row = node.querySelector(".doc-row");
      row.dataset.docId = doc.id;

      const icon = node.querySelector(".doc-icon");
      icon.classList.add(docIconClass(doc.file_type));
      icon.textContent = doc.file_type.toUpperCase().slice(0, 3);

      node.querySelector(".doc-name").textContent = doc.filename;
      const meta = node.querySelector(".doc-meta");
      meta.innerHTML = `${formatBytes(doc.file_size)} · ${formatDate(doc.upload_date)} · <span class="${statusClass(doc.status)}">${escapeHtml(statusLabel(doc))}</span>`;

      node.querySelector(".doc-delete").addEventListener("click", () => deleteDocument(doc.id, doc.filename));

      el.docList.appendChild(node);
    }
  }

  async function loadDocuments() {
    try {
      const data = await api("/api/documents");
      renderDocuments(data.documents);
    } catch (err) {
      console.error("Failed to load documents", err);
    }
  }

  async function deleteDocument(docId, filename) {
    if (!confirm(`Remove "${filename}" from your knowledge base? This can't be undone.`)) return;
    try {
      await api(`/api/documents/${docId}`, { method: "DELETE" });
      await loadDocuments();
      await refreshHealth();
    } catch (err) {
      alert(`Could not remove the document: ${err.message}`);
    }
  }

  function setUploadState(kind, message) {
    el.uploadInner.classList.toggle("is-processing", kind === "processing");
    if (kind === "processing") {
      el.uploadInner.innerHTML = `
        <p class="upload-title">${escapeHtml(message)}</p>
        <div class="progress-bar"><div class="progress-bar-fill"></div></div>
      `;
    } else {
      el.uploadInner.innerHTML = `
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 16V4M12 4L7 9M12 4L17 9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M4 16V18.5C4 19.8807 5.11929 21 6.5 21H17.5C18.8807 21 20 19.8807 20 18.5V16" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
        <p class="upload-title">Add a document</p>
        <p class="upload-hint">Drop a PDF, Word, or text file here, or <span class="upload-link">browse</span></p>
      `;
    }
  }

  async function uploadFile(file) {
    setUploadState("processing", `Processing ${file.name}…`);
    const formData = new FormData();
    formData.append("file", file);

    try {
      await api("/api/documents", { method: "POST", body: formData });
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setUploadState("idle");
      await loadDocuments();
      await refreshHealth();
    }
  }

  el.uploadZone.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", () => {
    if (el.fileInput.files.length) {
      uploadFile(el.fileInput.files[0]);
      el.fileInput.value = "";
    }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    el.uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.uploadZone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    el.uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.uploadZone.classList.remove("drag-over");
    })
  );
  el.uploadZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  // ---------------------------------------------------------------- chat

  function addMessage({ role, text, sources = [], time = new Date(), isError = false, isLoading = false }) {
    el.emptyState.style.display = "none";
    const node = el.tplMessage.content.cloneNode(true);
    const msgEl = node.querySelector(".message");
    msgEl.classList.add(role === "user" ? "from-user" : "from-assistant");
    if (isError) msgEl.classList.add("is-error");
    if (isLoading) msgEl.classList.add("is-loading");

    const textEl = node.querySelector(".message-text");
    if (isLoading) {
      textEl.innerHTML = `<span class="dot-flash"><span></span><span></span><span></span></span>`;
    } else if (role === "user") {
      textEl.textContent = text;
    } else {
      textEl.innerHTML = renderMarkdown(text);
    }

    if (sources.length) {
      const sourcesEl = node.querySelector(".message-sources");
      for (const src of sources) {
        const srcNode = el.tplSource.content.cloneNode(true);
        srcNode.querySelector(".source-name").textContent = src.filename || "Unknown source";
        srcNode.querySelector(".source-page").textContent = src.page_number ? `Page ${src.page_number}` : "";
        srcNode.querySelector(".source-relevance").textContent = `${Math.round(src.relevance * 100)}%`;
        sourcesEl.appendChild(srcNode);
      }
    }

    node.querySelector(".message-time").textContent = formatTime(time);

    el.messages.appendChild(node);
    el.messages.scrollTop = el.messages.scrollHeight;
    return el.messages.lastElementChild;
  }

  function autoResize() {
    el.questionInput.style.height = "auto";
    el.questionInput.style.height = Math.min(el.questionInput.scrollHeight, 140) + "px";
  }
  el.questionInput.addEventListener("input", autoResize);

  el.questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      el.composer.requestSubmit();
    }
  });

  async function askQuestion(question) {
    addMessage({ role: "user", text: question });
    state.history.push({ role: "user", text: question });
    state.lastQuestion = question;

    const loadingNode = addMessage({ role: "assistant", text: "", isLoading: true });
    el.sendBtn.disabled = true;

    try {
      const data = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: state.history.slice(0, -1) }),
      });
      loadingNode.remove();
      addMessage({ role: "assistant", text: data.answer, sources: data.sources || [] });
      state.history.push({ role: "assistant", text: data.answer });
    } catch (err) {
      loadingNode.remove();
      addMessage({ role: "assistant", text: err.message, isError: true });
    } finally {
      el.sendBtn.disabled = false;
    }
  }

  el.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = el.questionInput.value.trim();
    if (!question) return;
    el.questionInput.value = "";
    autoResize();
    askQuestion(question);
  });

  el.clearChatBtn.addEventListener("click", () => {
    if (!state.history.length) return;
    if (!confirm("Clear this conversation?")) return;
    state.history = [];
    el.messages.querySelectorAll(".message").forEach((m) => m.remove());
    el.emptyState.style.display = "flex";
  });

  // ---------------------------------------------------------------- init

  refreshHealth();
  loadDocuments();
  setInterval(refreshHealth, 20000);
})();
