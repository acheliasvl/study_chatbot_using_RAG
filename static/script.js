// ─── State ────────────────────────────────────────────────────────────────────
let activeConversationId  = null;
let isConversationPersisted = false;

// ─── DOM Refs ─────────────────────────────────────────────────────────────────
const chatContainer    = document.getElementById('chat-container');
const chatInput        = document.getElementById('chat-input');
const sendBtn          = document.getElementById('send-btn');
const pdfUpload        = document.getElementById('pdf-upload');
const newChatBtn       = document.getElementById('new-chat-btn');
const conversationList = document.getElementById('conversation-list');
const dateHeader       = document.getElementById('date-header');

// ─── Date ─────────────────────────────────────────────────────────────────────
dateHeader.textContent = new Date().toLocaleDateString('en-US', { year:'numeric', month:'short', day:'numeric' });

// ─── UUID helper ──────────────────────────────────────────────────────────────
function generateUUID() {
    return crypto.randomUUID ? crypto.randomUUID() :
        'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
            const r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function addMessage(text, isUser) {
    const div = document.createElement('div');
    div.classList.add('message', isUser ? 'user' : 'bot');
    div.textContent = text;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}

// ─── Pending conversation (UUID ready, NOT saved yet) ─────────────────────────
function initPendingConversation() {
    activeConversationId    = generateUUID();
    isConversationPersisted = false;
    chatContainer.innerHTML = '';
    highlightActive(null);
}

// ─── Persist on first action ──────────────────────────────────────────────────
async function ensurePersisted() {
    if (isConversationPersisted) return;
    const convCount = conversationList.querySelectorAll('.conv-btn').length;
    await fetch('/conversations/new', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ conversation_id: activeConversationId, name: `Chat ${convCount + 1}` })
    });
    isConversationPersisted = true;
    await loadConversations();
}

// ─── Conversations ────────────────────────────────────────────────────────────
async function loadConversations() {
    const res   = await fetch('/conversations');
    const convs = await res.json();
    renderConversationList(convs);
}

function renderConversationList(convs) {
    conversationList.innerHTML = '';
    convs.forEach(conv => {
        const btn = document.createElement('button');
        btn.className  = 'conv-btn icon-btn' + (conv.conversation_id === activeConversationId ? ' active' : '');
        btn.title      = conv.name;
        btn.dataset.id = conv.conversation_id;
        btn.innerHTML  = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span>${conv.name}</span>`;
        btn.addEventListener('click', () => selectConversation(conv.conversation_id));
        conversationList.appendChild(btn);
    });
}

function highlightActive(id) {
    document.querySelectorAll('.conv-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.id === id)
    );
}

function selectConversation(id) {
    activeConversationId    = id;
    isConversationPersisted = true;
    chatContainer.innerHTML = '';
    highlightActive(id);
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    await ensurePersisted();

    addMessage(text, true);
    chatInput.value = '';
    const botMsg = addMessage('', false);

    try {
        const response = await fetch('/chat', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ query: text, conversation_id: activeConversationId })
        });

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText  = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value);
            botMsg.textContent = fullText;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    } catch (e) {
        botMsg.textContent = 'Error connecting to server.';
    }
}

// ─── Upload ───────────────────────────────────────────────────────────────────
pdfUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    await ensurePersisted();

    addMessage(`Uploading ${file.name}...`, true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('conversation_id', activeConversationId);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result   = await response.json();
        addMessage(`✅ Uploaded! Added ${result.chunks} chunks to this conversation.`, false);
    } catch (e) {
        addMessage('Error uploading PDF.', false);
    }

    pdfUpload.value = '';
});

// ─── Events ───────────────────────────────────────────────────────────────────
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
newChatBtn.addEventListener('click', initPendingConversation);

// ─── Init ─────────────────────────────────────────────────────────────────────
(async () => {
    await loadConversations();
    initPendingConversation();
})();
