let mediaRecorder;
let audioChunks = [];

const recBtn = document.getElementById('recBtn');
const recText = document.getElementById('recText');
const sendBtn = document.getElementById('sendBtn');
const queryInput = document.getElementById('queryInput');
const resultBox = document.getElementById('resultBox');
const statusBadge = document.getElementById('statusBadge');
const queryDisplay = document.getElementById('queryDisplay');
const transcribedText = document.getElementById('transcribedText');

// Metric elements
const mTotal = document.getElementById('mTotal');
const mStt = document.getElementById('mStt');
const mRetr = document.getElementById('mRetr');
const mLlm = document.getElementById('mLlm');

recBtn.onclick = async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recBtn.classList.remove('recording');
        recText.innerText = 'Speak';
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio_file', audioBlob, 'mic.wav');
                sendQuery(formData);
            };
            
            mediaRecorder.start();
            recBtn.classList.add('recording');
            recText.innerText = 'Stop';
        } catch (err) {
            alert('Microphone access denied or unavailable: ' + err);
        }
    }
};

sendBtn.onclick = () => {
    const text = queryInput.value.trim();
    if (!text) return;
    const formData = new FormData();
    formData.append('query_text', text);
    sendQuery(formData);
};

queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendBtn.click();
});

async function sendQuery(formData) {
    setStatus('Processing...', 'processing');
    resultBox.innerText = "Executing RAG pipeline pipeline...";
    
    try {
        const res = await fetch('/query', { method: 'POST', body: formData });
        const data = await res.json();
        
        queryDisplay.style.display = 'block';
        transcribedText.innerText = data.query || 'N/A';
        resultBox.innerText = data.answer || 'No answer returned.';
        
        if (data.latency) {
            mTotal.innerText = `${data.latency.total_ms || 0} ms`;
            mStt.innerText = `${data.latency.stt_ms || 0} ms`;
            mRetr.innerText = `${data.latency.retrieval_ms || 0} ms`;
            mLlm.innerText = `${data.latency.llm_ms || 0} ms`;
        }
        
        setStatus(data.status || 'Success', data.status === 'error' ? 'error' : 'ready');
    } catch (err) {
        resultBox.innerText = "Error executing pipeline: " + err;
        setStatus('Error', 'error');
    }
}

function setStatus(text, type) {
    statusBadge.innerText = text;
    statusBadge.className = `badge ${type}`;
}