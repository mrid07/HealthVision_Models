// Global State
let isRunning = false;
let score = 0;
let pollingInterval = null;
let gameTimer = null;
let currentAnswer = null;
let questionTimeInitial = 10; // seconds
let questionTimeLeft = 10;
let questionTimer = null;

let sessionStartTime = 0;
let sessionHistory = {
    labels: [],
    hr: [],
    sys: [],
    dia: []
};

// Charts
let resultHrChart = null;
let resultBpChart = null;

// DOM Elements
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const nameInput = document.getElementById("userName");
const statusMsg = document.getElementById("statusMessage");
const videoFeed = document.getElementById("videoFeed");

const hrDisplay = document.getElementById("hrDisplay");
const rrDisplay = document.getElementById("rrDisplay");
const stressBox = document.getElementById("stressBox");
const stressValue = document.getElementById("stressValue");
const sqiValue = document.getElementById("sqiValue");

const questionBox = document.getElementById("questionBox");
const answerInput = document.getElementById("answerInput");
const feedbackBox = document.getElementById("mathFeedback");
const scoreVal = document.getElementById("scoreVal");
const timerFill = document.getElementById("timerFill");
const timeVal = document.getElementById("timeVal");

const resultsPanel = document.getElementById("resultsPanel");

// --- Event Listeners ---
startBtn.onclick = startSession;
stopBtn.onclick = stopSession;

answerInput.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        submitAnswer();
    }
});

// --- Session Management ---

async function startSession() {
    const user = nameInput.value.trim();
    if (!user) {
        alert("Please enter a participant name.");
        return;
    }

    statusMsg.innerText = "Initializing engine...";
    startBtn.disabled = true;
    answerInput.disabled = false;
    answerInput.focus();
    if(resultsPanel) resultsPanel.style.display = "none";

    try {
        const response = await fetch("/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_name: user })
        });
        const data = await response.json();

        if (data.status === "started") {
            isRunning = true;
            statusMsg.innerText = "Session Active";
            stopBtn.disabled = false;
            
            // Reset Data
            sessionStartTime = Date.now();
            sessionHistory = { labels: [], hr: [], sys: [], dia: [] };
            
            // Start Video Feed
            videoFeed.src = "/video_feed?" + new Date().getTime(); // timestamp to prevent caching
            
            // Start Polling Data
            pollingInterval = setInterval(fetchMetrics, 1000); // 1Hz update

            // Start Math Game
            score = 0;
            updateScore();
            nextQuestion();

        } else {
            statusMsg.innerText = "Failed to start engine.";
            startBtn.disabled = false;
        }
    } catch (e) {
        console.error(e);
        statusMsg.innerText = "Error connecting to server.";
        startBtn.disabled = false;
    }
}

async function stopSession() {
    statusMsg.innerText = "Stopping...";
    
    // Stop Game
    clearInterval(pollingInterval);
    clearInterval(questionTimer);
    isRunning = false;
    answerInput.disabled = true;

    try {
        await fetch("/stop", { method: "POST" });
        statusMsg.innerText = "Session Stopped";
    } catch (e) {
        statusMsg.innerText = "Error stopping session.";
    }

    // Reset UI
    videoFeed.src = "";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    questionBox.innerText = "? + ?";
    timerFill.style.width = "100%";
    
    // Show Charts
    renderResults();
}

// --- Data Polling ---

async function fetchMetrics() {
    if (!isRunning) return;

    try {
        const response = await fetch("/data");
        const data = await response.json();

        // Update UI
        hrDisplay.innerText = (data.hr !== null && data.hr !== undefined) ? data.hr.toFixed(1) : "--";
        rrDisplay.innerText = (data.rr !== null && data.rr !== undefined) ? data.rr.toFixed(1) : "--";
        sqiValue.innerText = (data.sqi !== null && data.sqi !== undefined) ? data.sqi.toFixed(2) : "--";

        // Collect History
        const timeLabel = Math.floor((Date.now() - sessionStartTime) / 1000) + "s";
        sessionHistory.labels.push(timeLabel);
        sessionHistory.hr.push(data.hr !== undefined ? data.hr : null);
        sessionHistory.sys.push(data.sys !== undefined ? data.sys : null);
        sessionHistory.dia.push(data.dia !== undefined ? data.dia : null);

        // Stress Logic (Visual)
        const stress = data.stress_label || "Unknown";
        stressValue.innerText = stress;
        
        if (stress.includes("High")) {
            stressBox.style.backgroundColor = "#ef4444"; // Red
        } else if (stress.includes("Moderate")) {
            stressBox.style.backgroundColor = "#f59e0b"; // Orange
        } else {
            stressBox.style.backgroundColor = "#10b981"; // Green (Calm/Unknown)
        }

    } catch (e) {
        console.error("Error fetching metrics", e);
    }
}

// --- Math Game Logic ---

function nextQuestion() {
    if (!isRunning) return;
    
    // Clear Input
    answerInput.value = "";
    feedbackBox.innerText = "";
    
    // Determine Difficulty based on time
    const elapsed = (Date.now() - sessionStartTime) / 1000;
    let operator, a, b;
    let timeLimit = 7.0;

    if (elapsed < 20) {
        // Level 1: Easy
        operator = Math.random() > 0.5 ? '+' : '-';
        a = Math.floor(Math.random() * 20) + 5;
        b = Math.floor(Math.random() * 15) + 2;
        timeLimit = 7.0;
    } else if (elapsed < 45) {
        // Level 2: Medium
        const dice = Math.random();
        if (dice < 0.33) {
            operator = '+';
            a = Math.floor(Math.random() * 50) + 20;
            b = Math.floor(Math.random() * 50) + 10;
        } else if (dice < 0.66) {
            operator = '-';
            a = Math.floor(Math.random() * 80) + 20;
            b = Math.floor(Math.random() * 40) + 5;
        } else {
            operator = '*';
            a = Math.floor(Math.random() * 10) + 2;
            b = Math.floor(Math.random() * 10) + 2;
        }
        timeLimit = 6.0;
    } else {
        // Level 3: Hard
        const dice = Math.random();
        if (dice < 0.4) {
            operator = '+';
            a = Math.floor(Math.random() * 200) + 50;
            b = Math.floor(Math.random() * 200) + 50;
        } else if (dice < 0.7) {
            operator = '-';
            a = Math.floor(Math.random() * 300) + 100;
            b = Math.floor(Math.random() * 150) + 50;
        } else {
            operator = '*';
            a = Math.floor(Math.random() * 15) + 3;
            b = Math.floor(Math.random() * 15) + 3;
        }
        timeLimit = 5.0; 
    }

    if (operator === '*') {
        currentAnswer = a * b;
    } else if (operator === '-') {
        if (a < b) [a, b] = [b, a];
        currentAnswer = a - b;
    } else {
        currentAnswer = a + b;
    }

    questionBox.innerText = `${a} ${operator} ${b} = ?`;
    
    // Start Timer
    startQuestionTimer(timeLimit);
}

function startQuestionTimer(limit) {
    clearInterval(questionTimer);
    questionTimeLeft = limit;
    const initialTime = limit;
    
    updateTimerUI(initialTime);

    questionTimer = setInterval(() => {
        questionTimeLeft -= 0.1;
        updateTimerUI(initialTime);

        if (questionTimeLeft <= 0) {
            handleTimeout();
        }
    }, 100);
}

function updateTimerUI(initialTime) {
    const percentage = (questionTimeLeft / initialTime) * 100;
    timerFill.style.width = `${percentage}%`;
    timeVal.innerText = Math.ceil(questionTimeLeft) + "s";
    
    // Change color as it gets low
    if (percentage < 30) {
        timerFill.style.backgroundColor = "#ef4444";
    } else {
        timerFill.style.backgroundColor = "#22c55e";
    }
}

function submitAnswer() {
    if (!isRunning) return;

    const val = parseInt(answerInput.value);
    
    if (isNaN(val)) return; // ignore empty

    if (val === currentAnswer) {
        // Correct
        feedbackBox.innerText = "Correct!";
        feedbackBox.className = "task-feedback correct";
        score += 10;
        updateScore();
        nextQuestion();
    } else {
        // Wrong
        feedbackBox.innerText = "Wrong!";
        feedbackBox.className = "task-feedback wrong";
        score -= 5;
        updateScore();
        nextQuestion();
    }
}

function handleTimeout() {
    feedbackBox.innerText = "Time Up!";
    feedbackBox.className = "task-feedback wrong";
    score -= 2;
    updateScore();
    nextQuestion();
}

function updateScore() {
    scoreVal.innerText = score;
}

// --- Charting ---

function renderResults() {
    if(resultsPanel) resultsPanel.style.display = "block";
    
    // Destroy existing charts if any
    if (resultHrChart) resultHrChart.destroy();
    if (resultBpChart) resultBpChart.destroy();
    
    // HR Chart
    const ctxHr = document.getElementById('resultHrChart').getContext('2d');
    resultHrChart = new Chart(ctxHr, {
        type: 'line',
        data: {
            labels: sessionHistory.labels,
            datasets: [{
                label: 'Heart Rate (bpm)',
                data: sessionHistory.hr,
                borderColor: '#f43f5e',
                backgroundColor: 'rgba(244, 63, 94, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#334155' } },
                y: { grid: { color: '#334155' } }
            }
        }
    });

    // BP Chart
    const ctxBp = document.getElementById('resultBpChart').getContext('2d');
    resultBpChart = new Chart(ctxBp, {
        type: 'line',
        data: {
            labels: sessionHistory.labels,
            datasets: [
                {
                    label: 'Systolic',
                    data: sessionHistory.sys,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Diastolic',
                    data: sessionHistory.dia,
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#334155' } },
                y: { grid: { color: '#334155' } }
            }
        }
    });

     // Scroll to results
     if(resultsPanel) resultsPanel.scrollIntoView({ behavior: 'smooth' });

     // Calculate Averages
     const validHr = sessionHistory.hr.filter(x => x !== null && x !== undefined && !isNaN(x));
     const validSys = sessionHistory.sys.filter(x => x !== null && x !== undefined && !isNaN(x));
     const validDia = sessionHistory.dia.filter(x => x !== null && x !== undefined && !isNaN(x));
 
     const avgHr = validHr.length ? (validHr.reduce((a, b) => a + b, 0) / validHr.length).toFixed(1) : "--";
     const avgSys = validSys.length ? (validSys.reduce((a, b) => a + b, 0) / validSys.length).toFixed(0) : "--";
     const avgDia = validDia.length ? (validDia.reduce((a, b) => a + b, 0) / validDia.length).toFixed(0) : "--";
 
     // Update DOM
     const avgHrEl = document.getElementById("avgHrDisplay");
     const avgBpEl = document.getElementById("avgBpDisplay");
     
     if(avgHrEl) avgHrEl.innerText = avgHr;
     if(avgBpEl) avgBpEl.innerText = `${avgSys}/${avgDia}`;
}
