// Simple AI Tutor Web UI
const API = '/api';
let currentModel = 'base';

// === Initialize ===
document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupListeners();
    await loadCurrentModel();
}

function setupListeners() {
    // Chat
    document.getElementById('send-btn').onclick = sendMessage;
    document.getElementById('message-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Model
    document.getElementById('model-switch-btn').onclick = switchModel;

    // Documents
    document.getElementById('upload-btn').onclick = uploadFiles;
    document.getElementById('file-input').onchange = (e) => {
        const files = Array.from(e.target.files).map(f => f.name).join(', ');
        document.getElementById('selected-files').innerHTML = files ? `Selected: ${files}` : '';
    };

    // Session
    document.getElementById('reset-btn').onclick = resetSession;
    document.getElementById('stats-btn').onclick = showStats;
}

// === Chat Functions ===
async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (!message) return;

    addMessage('user', message);
    input.value = '';

    const btn = document.getElementById('send-btn');
    btn.disabled = true;
    btn.textContent = 'Sending...';

    try {
        const res = await fetch(`${API}/chat/message`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message})
        });
        const data = await res.json();
        addMessage('bot', data.response, data.sources);
    } catch (err) {
        addMessage('system', 'Error: ' + err.message);
    }

    btn.disabled = false;
    btn.textContent = 'Send';
}

function addMessage(role, text, sources = []) {
    const chat = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = `message message-${role}`;

    const content = document.createElement('div');
    content.className = 'message-content';

    // Convert Markdown → HTML
    let html = text;

    if (window.marked) {
        html = marked.parse(text);
    }

    // Sanitize HTML (important for security)
    if (window.DOMPurify) {
        html = DOMPurify.sanitize(html);
    }

    content.innerHTML = html;
    div.appendChild(content);

    if (sources.length > 0) {
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = `Sources: ${sources.length} document(s)`;
        details.appendChild(summary);

        sources.forEach((s, i) => {
            const p = document.createElement('p');
            p.textContent = `${i + 1}. ${s.text.substring(0, 150)}...`;
            details.appendChild(p);
        });

        div.appendChild(details);
    }

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

// === Model Functions ===
async function loadCurrentModel() {
    try {
        const res = await fetch(`${API}/models/current`);
        const data = await res.json();
        currentModel = data.model_type;

        document.querySelector(`input[value="${data.model_type}"]`).checked = true;
        document.getElementById('model-info').innerHTML =
            `<strong>Current:</strong> ${formatModelName(data.model_type)}<br>` +
            `<strong>RAG:</strong> ${data.rag_enabled ? 'Enabled' : 'Disabled'}`;
    } catch (err) {
        console.error('Load model error:', err);
    }
}

async function switchModel() {
    const selected = document.querySelector('input[name="model-type"]:checked').value;
    if (selected === currentModel) {
        alert('Already using this model');
        return;
    }

    showLoading(`Loading ${selected} model...`);

    try {
        const res = await fetch(`${API}/models/switch`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_type: selected})
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || 'Switch failed');
        }

        const data = await res.json();

        if (data.success) {
            currentModel = selected;
            await loadCurrentModel();
            alert(`Switched to ${selected} (${data.load_time_seconds.toFixed(1)}s)`);
            addMessage('system', `Model: ${formatModelName(selected)} (RAG: ${data.rag_enabled ? 'on' : 'off'})`);
        }
    } catch (err) {
        alert('Switch failed: ' + err.message);
    }

    hideLoading();
}

function formatModelName(type) {
    const names = {'base': 'Base Model', 'base_rag': 'Base + RAG', 'finetuned': 'Fine-tuned'};
    return names[type] || type;
}

// === Document Functions ===
async function uploadFiles() {
    const input = document.getElementById('file-input');
    const files = input.files;

    if (files.length === 0) {
        alert('Select files first');
        return;
    }

    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }

    showLoading('Uploading...');

    try {
        const res = await fetch(`${API}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.success) {
            alert(`Uploaded ${data.uploaded_files.length} file(s)`);
            input.value = '';
            document.getElementById('selected-files').innerHTML = '';
            await refreshFileList();
            addMessage('system', `Uploaded ${data.uploaded_files.length} document(s)`);
        }
    } catch (err) {
        alert('Upload failed: ' + err.message);
    }

    hideLoading();
}

async function refreshFileList() {
    try {
        const res = await fetch(`${API}/documents/list`);
        const data = await res.json();

        const div = document.getElementById('uploaded-files-content');
        if (data.documents.length === 0) {
            div.innerHTML = '<p>No files uploaded yet</p>';
        } else {
            div.innerHTML = data.documents.map(d =>
                `<div>📄 ${d.filename} (${formatBytes(d.size_bytes)})</div>`
            ).join('');
        }
    } catch (err) {
        console.error('Refresh list error:', err);
    }
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// === Session Functions ===
async function resetSession() {
    if (!confirm('Reset session and clear history?')) return;

    showLoading('Resetting...');

    try {
        await fetch(`${API}/session/reset`, {method: 'POST'});
        document.querySelectorAll('.message:not(.message-system)').forEach(m => m.remove());
        addMessage('system', 'Session reset. Start a new conversation!');
    } catch (err) {
        alert('Reset failed');
    }

    hideLoading();
}

async function showStats() {
    showLoading('Loading stats...');

    try {
        const res = await fetch(`${API}/session/stats`);
        const stats = await res.json();

        // API nests dialogue data under dialogue.conversation and dialogue.lesson
        const conversation = stats.dialogue?.conversation || {};
        const lesson       = stats.dialogue?.lesson       || {};
        const mem          = stats.llm_memory             || {};
        const rag          = stats.rag                    || {};
        const assess       = stats.assessment             || {};

        // Intent distribution — sorted by count descending
        const intents = conversation.intent_distribution || {};
        const intentRows = Object.entries(intents)
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
            .join('');

        // Answer accuracy
        const correct   = lesson.correct_answers   || 0;
        const incorrect = lesson.incorrect_answers || 0;
        const total     = correct + incorrect;
        const accuracy  = total > 0 ? Math.round((correct / total) * 100) + '%' : '—';

        // GPU memory bar (filled ratio of reserved)
        const allocGb   = mem.allocated_gb   ? mem.allocated_gb.toFixed(1)   : null;
        const reservedGb= mem.reserved_gb    ? mem.reserved_gb.toFixed(1)    : null;
        const memLine   = allocGb
            ? `${allocGb} GB allocated / ${reservedGb} GB reserved`
            : 'N/A';

        const modal   = document.getElementById('stats-modal');
        const content = document.getElementById('stats-content');

        content.innerHTML = `
            <h3>Conversation</h3>
            <table>
                <tr><td>Turns</td><td>${conversation.total_turns || 0}</td></tr>
                <tr><td>Duration</td><td>${conversation.session_duration_minutes != null ? conversation.session_duration_minutes.toFixed(1) + ' min' : '—'}</td></tr>
            </table>

            ${intentRows ? `
            <h3>Intent breakdown</h3>
            <table>
                <tr><th>Intent</th><th>Count</th></tr>
                ${intentRows}
            </table>` : ''}

            <h3>Learning progress</h3>
            <table>
                <tr><td>Topic</td><td>${lesson.topic && lesson.topic !== 'general' ? lesson.topic : '—'}</td></tr>
                <tr><td>Phase</td><td>${lesson.phase || '—'}</td></tr>
                ${lesson.current_question ? `<tr><td>Active question</td><td style="font-style:italic">${lesson.current_question.substring(0, 80)}${lesson.current_question.length > 80 ? '…' : ''}</td></tr>` : ''}
                <tr><td>Correct answers</td><td>${correct}</td></tr>
                <tr><td>Incorrect answers</td><td>${incorrect}</td></tr>
                <tr><td>Accuracy</td><td>${accuracy}</td></tr>
                <tr><td>Hints used</td><td>${lesson.hints_used || 0}</td></tr>
                <tr><td>Attempts</td><td>${lesson.attempts || 0}</td></tr>
            </table>

            ${rag.num_documents != null ? `
            <h3>Knowledge base</h3>
            <table>
                <tr><td>Documents</td><td>${rag.num_documents}</td></tr>
                <tr><td>Vectors</td><td>${rag.num_vectors || rag.num_documents}</td></tr>
                <tr><td>Embedding dim</td><td>${rag.dimension || '—'}</td></tr>
                <tr><td>Index</td><td>${rag.index_type || '—'}</td></tr>
            </table>` : ''}

            ${mem.model_loaded != null ? `
            <h3>Model</h3>
            <table>
                <tr><td>Device</td><td>${mem.device || '—'}</td></tr>
                <tr><td>GPU memory</td><td>${memLine}</td></tr>
                <tr><td>Error analyzer</td><td>${assess.components_active?.error_analyzer ? 'active' : 'off'}</td></tr>
                <tr><td>Hint generator</td><td>${assess.components_active?.hint_generator ? 'active' : 'off'}</td></tr>
            </table>` : ''}
        `;

        modal.classList.add('active');
        modal.onclick = (e) => {
            if (e.target === modal || e.target.classList.contains('modal-close')) {
                modal.classList.remove('active');
            }
        };
    } catch (err) {
        alert('Stats failed');
    }

    hideLoading();
}

// === UI Helpers ===
function showLoading(msg = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-message').textContent = msg;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

// Initialize file list
refreshFileList();
