// app.js — CAD frontend logic

const SAMPLES = {
  'deep-bio': {
    subject: 'Biology',
    question: 'Explain how photosynthesis works and why it is important to life on Earth.',
    answer: 'Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. It occurs in two stages: the light-dependent reactions in the thylakoid membranes produce ATP and NADPH while splitting water and releasing oxygen, and the Calvin cycle in the stroma fixes CO2 into glucose. The equation is 6CO2 + 6H2O + light → C6H12O6 + 6O2. Photosynthesis is foundational to life because it forms the base of almost every food chain, regulates atmospheric oxygen and CO2, and drives the carbon cycle.'
  },
  'shallow-bio': {
    subject: 'Biology',
    question: 'Explain how photosynthesis works and why it is important to life on Earth.',
    answer: 'Photosynthesis is when plants use sunlight to make food. They take in CO2 and water and make glucose and oxygen. It is important because plants give us oxygen to breathe.'
  },
  'deep-phys': {
    subject: 'Physics',
    question: "Explain Newton's second law of motion and give a real-world example.",
    answer: "Newton's second law states that the net force on an object equals mass times acceleration (F=ma). The law applies to net force — meaning the vector sum of all forces determines acceleration. A heavier car requires proportionally more force to achieve the same acceleration, which is why sports cars have high power-to-weight ratios. Seatbelts use this law: in a sudden deceleration, F = m·Δv/Δt shows that increasing the time interval reduces the peak force on the passenger."
  },
  'shallow-phys': {
    subject: 'Physics',
    question: "Explain Newton's second law of motion and give a real-world example.",
    answer: "Newton's second law is F=ma. Force equals mass times acceleration. An example is pushing a box — if you push harder it goes faster."
  }
};

const session = { n: 0, deep: 0, gusSum: 0 };

function showPage(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function loadSample(key) {
  const s = SAMPLES[key];
  document.getElementById('subject').value = s.subject;
  document.getElementById('question').value = s.question;
  document.getElementById('answer').value = s.answer;
}

function clearForm() {
  ['question', 'answer', 'time_spent', 'edits'].forEach(id => document.getElementById(id).value = '');
  document.getElementById('result-card').classList.remove('vis');
}

async function runAnalysis() {
  const answer = document.getElementById('answer').value.trim();
  const question = document.getElementById('question').value.trim();
  if (!answer) { alert('Please provide a student answer.'); return; }

  const btn = document.getElementById('abtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analysing…';

  const lb = document.getElementById('lb');
  const lf = document.getElementById('lf');
  const lm = document.getElementById('lm');
  lb.classList.add('vis'); lm.style.display = 'block';

  const msgs = ['Preprocessing text…', 'Extracting features…', 'Running classifier…', 'Calibrating GUS…'];
  let prog = 0;
  const iv = setInterval(() => {
    prog = Math.min(prog + 12, 88);
    lf.style.width = prog + '%';
    lm.textContent = msgs[Math.min(Math.floor(prog / 25), 3)];
  }, 300);

  try {
    const res = await fetch('/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        answer,
        question,
        subject: document.getElementById('subject').value,
        grade: document.getElementById('grade').value,
        time_spent: document.getElementById('time_spent').value || null,
        edits: document.getElementById('edits').value || null,
      })
    });
    const data = await res.json();
    clearInterval(iv); lf.style.width = '100%';

    if (data.error) { alert('Error: ' + data.error); return; }

    setTimeout(() => {
      lb.classList.remove('vis'); lf.style.width = '0%'; lm.style.display = 'none';
      btn.disabled = false;
      btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="6.5" cy="6.5" r="4"/><path d="M10.5 10.5L14 14"/></svg> Analyse answer';
      showResult(data);
    }, 300);

  } catch (e) {
    clearInterval(iv); lb.classList.remove('vis'); lm.style.display = 'none';
    btn.disabled = false;
    btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="6.5" cy="6.5" r="4"/><path d="M10.5 10.5L14 14"/></svg> Analyse answer';
    alert('Request failed: ' + e.message);
  }
}

function showResult(r) {
  const rc = document.getElementById('result-card');
  rc.classList.add('vis');

  const gb = document.getElementById('gus-block');
  gb.className = 'gus-block ' + (r.verdict === 'Deep Understanding' ? 'vd' : r.verdict === 'Shallow Understanding' ? 'vs' : 'vm');

  document.getElementById('gus-num').textContent = r.gus;
  document.getElementById('v-big').textContent = r.verdict;
  document.getElementById('v-note').textContent = 'GUS score: ' + r.gus + '/100 · Confidence: ' + Math.round(r.calibrated_prob * 100) + '%';

  const vb = document.getElementById('v-badge');
  vb.textContent = r.verdict;
  vb.className = 'badge';
  if (r.verdict === 'Deep Understanding') vb.classList.add('badge-green');
  else if (r.verdict === 'Shallow Understanding') vb.style.cssText = 'background:var(--warn-light);color:var(--warn)';
  else vb.style.cssText = 'background:var(--gold-light);color:var(--gold)';

  const vals = [r.semantic_depth, r.conceptual_links, r.confidence_score, r.lexical_richness];
  vals.forEach((v, i) => {
    document.getElementById('m' + i).textContent = (v || 0) + '/100';
    document.getElementById('b' + i).style.width = (v || 0) + '%';
  });

  document.getElementById('signals').innerHTML = (r.signals || []).map(s =>
    `<div class="sig-row"><div class="sig-dot ${s.type === 'positive' ? 'sp' : s.type === 'negative' ? 'sn' : 'su'}"></div><span>${s.text}</span></div>`
  ).join('');

  document.getElementById('rec').textContent = r.recommendation || '';

  session.n++;
  if (r.verdict === 'Deep Understanding') session.deep++;
  session.gusSum += r.gus;
  document.getElementById('s-count').textContent = session.n;
  document.getElementById('s-deep').textContent = Math.round(session.deep / session.n * 100) + '%';
  document.getElementById('s-avg').textContent = Math.round(session.gusSum / session.n);

  rc.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
