const chatContainer = document.getElementById('chat-container');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');
const pdfUpload = document.getElementById('pdf-upload');

function addMessage(text, isUser) {
    const div = document.createElement('div');
    div.classList.add('message', isUser ? 'user' : 'bot');
    div.textContent = text;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage(text, true);
    chatInput.value = '';

    const botMsg = addMessage('', false);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value);
            botMsg.textContent = fullText;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    } catch (e) {
        botMsg.textContent = "Error connecting to server.";
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

clearBtn.addEventListener('click', () => {
    chatContainer.innerHTML = '';
});

pdfUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    addMessage(`Uploading ${file.name}...`, true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        addMessage(`Success! Added ${result.chunks} chunks to database.`, false);
    } catch (e) {
        addMessage(`Error uploading PDF.`, false);
    }
});
