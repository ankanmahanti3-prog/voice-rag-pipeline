// ===== Aira — voice RAG frontend =====

// --- DOM refs ---
const orbWrap = document.getElementById('orbWrap');
const hexOrb = document.getElementById('hexOrb');
const hexSvg = document.getElementById('hexSvg');
const hexPolyEl = document.getElementById('hexPoly');
const hexDotsG = document.getElementById('hexDots');
const barsEl = document.getElementById('bars');
const barEls = Array.from(barsEl.querySelectorAll('.bar'));

const minDot = document.getElementById('minDot');
const minStatus = document.getElementById('minStatus');
const hint = document.getElementById('hint');
const voiceCenter = document.getElementById('voiceCenter');

const transcriptEl = document.getElementById('transcript');

const answerHeadline = document.getElementById('answerHeadline');
const answerKickerText = document.getElementById('answerKickerText');
const metaRow = document.getElementById('metaRow');
const statusPill = document.getElementById('statusPill');
const latencyPill = document.getElementById('latencyPill');
const sourcesWrap = document.getElementById('sourcesWrap');
const sourcesHead = document.getElementById('sourcesHead');
const sourcesList = document.getElementById('sourcesList');

const mStt = document.getElementById('m-stt');
const mRet = document.getElementById('m-ret');
const mLlm = document.getElementById('m-llm');
const mTotal = document.getElementById('m-total');

const DEFAULT_HEADLINE = 'Ask a question or press the orb to speak — aira will listen, retrieve, and answer.';

// --- build the hexagon geometry ---
const dots = [];
const finalPositions = [];
hexPolyEl.setAttribute('points', Array.from({ length: 6 }, (_, i) => {
    const a = Math.PI / 3 * i - Math.PI / 6;
    return `${50 + Math.cos(a) * 10},${50 + Math.sin(a) * 10}`;
}).join(' '));

for (let i = 0; i < 6; i++) {
    const ang = i * 60 * Math.PI / 180;
    const r = 30;
    const cx = 50 + Math.cos(ang) * r;
    const cy = 50 + Math.sin(ang) * r;
    finalPositions.push({ cx, cy });
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', cx);
    c.setAttribute('cy', cy);
    c.setAttribute('r', 9.5);
    c.setAttribute('class', 'hex-dot');
    c.style.transformOrigin = `${cx}px ${cy}px`;
    c.style.transition = 'all 740ms cubic-bezier(0.16,1,0.3,1)';
    hexDotsG.appendChild(c);
    dots.push(c);
}

// --- state machine ---
let state = 'idle'; // idle | listening | retrieving | complete | error
let audioCtx, analyser, dataArray, micSource, raf, mediaStream, breathRaf, recognition;
let mediaRecorder, audioChunks = [];

function setState(s) {
    state = s;
    minDot.className = 'min-dot ' + s;
    orbWrap.classList.toggle('listening', s === 'listening');
    voiceCenter.classList.toggle('listening', s === 'listening');

    if (s === 'idle') { minStatus.textContent = 'idle'; hint.style.opacity = '0'; }
    if (s === 'listening') { minStatus.textContent = 'listening'; hint.style.opacity = '1'; }
    if (s === 'retrieving') { minStatus.textContent = 'retrieving'; hint.style.opacity = '0'; }
    if (s === 'complete') { minStatus.textContent = 'complete'; }
    if (s === 'error') { minStatus.textContent = 'error'; }
}

// idle breathing animation for the hex orb
function idleBreath() {
    if (state !== 'idle') return;
    const t = Date.now() / 1000;
    dots.forEach((el, i) => { el.style.transform = `scale(${1 + Math.sin(t * 1.1 + i * 0.7) * 0.035})`; });
    hexSvg.style.transform = `rotate(${Math.sin(t * 0.28) * 0.7}deg)`;
    breathRaf = requestAnimationFrame(idleBreath);
}
idleBreath();

// live mic-reactive bars
function animateLive() {
    if (!analyser) return;
    analyser.getByteFrequencyData(dataArray);
    const step = Math.floor(dataArray.length / 4);
    for (let i = 0; i < 4; i++) {
        let sum = 0;
        for (let j = 0; j < step; j++) sum += dataArray[i * step + j];
        const avg = sum / step;
        const targetH = 19 + (avg / 255) * 40;
        const targetFill = 12 + (avg / 255) * 70;
        const curH = parseFloat(barEls[i].dataset.h || '21');
        const curF = parseFloat(barEls[i].dataset.f || '25');
        const nh = curH + (targetH - curH) * 0.44;
        const nf = curF + (targetFill - curF) * 0.44;
        barEls[i].dataset.h = nh; barEls[i].dataset.f = nf;
        barEls[i].style.height = nh.toFixed(1) + 'px';
        barEls[i].style.setProperty('--fill', nf.toFixed(1) + '%');
        barEls[i].classList.add('filled');
    }
    raf = requestAnimationFrame(animateLive);
}

// the orb "collapse back to hex" transition when recording stops
function playStopAnimation() {
    barEls.forEach((b, i) => {
        b.style.transition = `all 380ms cubic-bezier(0.16,1,0.3,1) ${i * 28}ms`;
        b.style.transform = 'scale(0.58)'; b.style.opacity = '0';
    });
    barsEl.style.transition = 'all 440ms cubic-bezier(0.16,1,0.3,1)';
    barsEl.style.transform = 'translate(-50%,-50%) scale(0.82)'; barsEl.style.opacity = '0';

    dots.forEach((el) => {
        el.style.transition = 'none';
        el.setAttribute('cx', 50); el.setAttribute('cy', 50); el.setAttribute('r', 11);
        el.style.fill = '#0C0C0C'; el.style.stroke = 'none'; el.style.opacity = '0';
        el.style.transform = 'scale(0.32)'; el.style.filter = 'blur(1px)';
    });
    hexPolyEl.style.opacity = '0'; hexPolyEl.style.transform = 'scale(0.7)';
    hexPolyEl.style.transformOrigin = '50px 50px'; hexPolyEl.style.transition = 'none';
    void hexSvg.getBoundingClientRect();

    dots.forEach((el, i) => {
        const final = finalPositions[i];
        setTimeout(() => {
            el.style.transition = `cx 700ms cubic-bezier(0.16,1,0.3,1), cy 700ms cubic-bezier(0.16,1,0.3,1), r 540ms cubic-bezier(0.16,1,0.3,1), fill 360ms ease 160ms, stroke 360ms ease 160ms, opacity 420ms ease, transform 700ms cubic-bezier(0.16,1,0.3,1), filter 480ms ease`;
            el.setAttribute('cx', final.cx); el.setAttribute('cy', final.cy); el.setAttribute('r', 9.5);
            el.style.fill = '#FFFFFF'; el.style.stroke = '#0C0C0C'; el.style.opacity = '1';
            el.style.transform = 'scale(1)'; el.style.filter = 'blur(0px)';
        }, 90 + i * 54);
    });
    setTimeout(() => {
        hexPolyEl.style.transition = 'all 540ms cubic-bezier(0.16,1,0.3,1)';
        hexPolyEl.style.opacity = '0.22'; hexPolyEl.style.transform = 'scale(1)';
    }, 300);
}

function resetOrbForListening() {
    barEls.forEach(b => { b.style.transition = ''; b.style.transform = ''; b.style.opacity = ''; });
    barsEl.style.transition = ''; barsEl.style.transform = ''; barsEl.style.opacity = '';
}

// --- recording ---
async function startListening() {
    if (state !== 'idle') return;
    resetOrbForListening();
    setState('listening');
    if (breathRaf) cancelAnimationFrame(breathRaf);
    clearTranscript(true);
    hideAnswerMeta();

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // visual analyser
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256; analyser.smoothingTimeConstant = 0.78;
        micSource = audioCtx.createMediaStreamSource(mediaStream);
        micSource.connect(analyser);
        dataArray = new Uint8Array(analyser.frequencyBinCount);
        animateLive();

        // actual recording for STT upload
        audioChunks = [];
        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'mic.wav');
            submitQuery(formData);
        };
        mediaRecorder.start();

        // optional cosmetic live captions
        startLiveCaptions();
    } catch (err) {
        console.warn('Microphone unavailable:', err);
        if (raf) cancelAnimationFrame(raf);
        setState('error');
        setAnswer("Couldn't access your microphone — check your browser's mic permissions and try again.");
        setTimeout(() => { setState('idle'); idleBreath(); }, 2400);
    }
}

function stopListening() {
    if (state !== 'listening') return;
    if (raf) cancelAnimationFrame(raf);
    if (recognition) { try { recognition.stop(); } catch (e) {} }
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    if (audioCtx) { try { audioCtx.close(); } catch (e) {} audioCtx = null; analyser = null; }

    setState('retrieving');
    playStopAnimation();

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop(); // triggers onstop -> submitQuery
    } else {
        endOrbCycle(); // nothing was actually recorded
    }
}

function startLiveCaptions() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { transcriptEl.innerHTML = '<span>listening…</span>'; return; }
    recognition = new SR();
    recognition.continuous = true; recognition.interimResults = true; recognition.lang = 'en-US';
    let finalT = '';
    recognition.onresult = (e) => {
        let interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
            if (e.results[i].isFinal) finalT += e.results[i][0].transcript + ' ';
            else interim += e.results[i][0].transcript;
        }
        transcriptEl.innerHTML = '<span>' + (finalT + interim || 'listening…') + '</span>';
    };
    recognition.onend = () => { if (state === 'listening') try { recognition.start(); } catch (e) {} };
    try { recognition.start(); } catch (e) {}
}

function endOrbCycle() {
    setState('complete');
    setTimeout(() => {
        barEls.forEach(b => {
            b.style.transition = 'none'; b.style.height = '22px';
            b.style.setProperty('--fill', '26%'); b.style.transform = 'scale(1)';
            b.style.opacity = '1'; b.classList.remove('filled');
        });
        barsEl.style.transition = 'none';
        barsEl.style.transform = 'translate(-50%,-50%) scale(0.88)'; barsEl.style.opacity = '0';
        void barsEl.getBoundingClientRect();
        barEls.forEach(b => { b.style.transition = ''; });
        setState('idle');
        idleBreath();
    }, 900);
}

// --- transcript / answer helpers ---
function clearTranscript(listeningPlaceholder) {
    transcriptEl.innerHTML = listeningPlaceholder
        ? '<span class="placeholder">listening…</span>'
        : '<span class="placeholder">your words will appear here...</span>';
}

function hideAnswerMeta() {
    metaRow.style.display = 'none';
    sourcesWrap.style.display = 'none';
}

function setAnswer(text) {
    answerHeadline.classList.add('updating');
    setTimeout(() => {
        answerHeadline.textContent = text;
        answerHeadline.classList.remove('updating');
    }, 120);
}

function renderMetrics(latency) {
    if (!latency) return;
    mStt.textContent = latency.stt_ms != null ? `${latency.stt_ms}ms` : '--';
    mRet.textContent = latency.retrieval_ms != null ? `${latency.retrieval_ms}ms` : '--';
    mLlm.textContent = latency.llm_ms != null ? `${latency.llm_ms}ms` : '--';
    mTotal.textContent = latency.total_ms != null ? `${latency.total_ms}ms` : '--';
}

function renderSources(sources) {
    if (!Array.isArray(sources) || sources.length === 0) {
        sourcesWrap.style.display = 'none';
        return;
    }
    sourcesHead.textContent = `Sources — ${sources.length} retrieved`;
    sourcesList.innerHTML = '';
    sources.forEach((src, i) => {
        const row = document.createElement('div');
        row.className = 'source-row';
        const num = String(i + 1).padStart(2, '0');
        row.innerHTML = `
            <div class="source-num mono">${num}</div>
            <div>
                <div class="source-title">${escapeHtml(src.title || src.name || 'Untitled source')}</div>
                <div class="source-meta mono">${escapeHtml(src.meta || src.subtitle || '')}</div>
            </div>`;
        sourcesList.appendChild(row);
        if (i < sources.length - 1) {
            const div = document.createElement('div');
            div.className = 'divider';
            sourcesList.appendChild(div);
        }
    });
    sourcesWrap.style.display = 'block';
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

// --- network ---
async function submitQuery(formData) {
    answerKickerText.textContent = 'Response';
    try {
        const res = await fetch('/query', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.query) transcriptEl.innerHTML = `<span>${escapeHtml(data.query)}</span>`;
        else clearTranscript(false);
        setAnswer(data.answer || 'No answer returned.');
        renderMetrics(data.latency);
        renderSources(data.sources);

        const isError = data.status === 'error';
        statusPill.textContent = data.status || (isError ? 'error' : 'ready');
        statusPill.classList.toggle('error', isError);
        latencyPill.textContent = data.latency && data.latency.total_ms != null
            ? `${data.latency.total_ms}ms total`
            : '—';
        metaRow.style.display = 'flex';

        setState(isError ? 'error' : 'complete');
    } catch (err) {
        setAnswer('Something went wrong reaching the pipeline: ' + err.message);
        statusPill.textContent = 'error';
        statusPill.classList.add('error');
        latencyPill.textContent = '—';
        metaRow.style.display = 'flex';
        setState('error');
    } finally {
        setTimeout(() => {
            barEls.forEach(b => {
                b.style.transition = 'none'; b.style.height = '22px';
                b.style.setProperty('--fill', '26%'); b.style.transform = 'scale(1)';
                b.style.opacity = '1'; b.classList.remove('filled');
            });
            barsEl.style.transition = 'none';
            barsEl.style.transform = 'translate(-50%,-50%) scale(0.88)'; barsEl.style.opacity = '0';
            void barsEl.getBoundingClientRect();
            barEls.forEach(b => { b.style.transition = ''; });
            setState('idle');
            idleBreath();
        }, 900);
    }
}

// --- orb interactions ---
orbWrap.addEventListener('click', () => {
    if (state === 'idle') startListening();
    else if (state === 'listening') stopListening();
});
orbWrap.addEventListener('keydown', (e) => {
    if (e.code === 'Enter' || e.code === 'Space') {
        e.preventDefault();
        if (state === 'idle') startListening();
        else if (state === 'listening') stopListening();
    }
});
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && document.activeElement !== orbWrap) {
        e.preventDefault();
        if (state === 'idle') startListening();
        else if (state === 'listening') stopListening();
    }
});

setState('idle');