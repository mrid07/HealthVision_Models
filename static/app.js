
const videoFeed = document.getElementById("videoFeed");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const nameInput = document.getElementById("userName");
const feedback = document.getElementById("feedback");

// Session tracking for averages
let sessionHistory = {
    hr: [],
    sys: [],
    dia: [],
    sys_new: [],
    dia_new: []
};

// --- Charts Initialization ---
const MAX_POINTS = 60; // Keep last 60 points

let hrChartCtx = document.getElementById('hrChart').getContext('2d');
let hrChart = new Chart(hrChartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Heart Rate (bpm)',
            data: [],
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { 
                display: false,
                grid: { color: '#334155' }
            },
            y: { 
                beginAtZero: false,
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' } 
            }
        },
        plugins: {
            legend: { labels: { color: '#e2e8f0' } }
        },
        animation: false
    }
});

let bpChartCtx = document.getElementById('bpChart').getContext('2d');
let bpChart = new Chart(bpChartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Systolic (mmHg)',
                data: [],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0)',
                borderWidth: 2,
                tension: 0.4
            },
            {
                label: 'Diastolic (mmHg)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0)',
                borderWidth: 2,
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { 
                display: false,
                grid: { color: '#334155' }
            },
            y: { 
                beginAtZero: false,
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' }
            }
        },
        plugins: {
            legend: { labels: { color: '#e2e8f0' } }
        },
        animation: false
    }
});

let bpNewChartCtx = document.getElementById('bpNewChart').getContext('2d');
let bpNewChart = new Chart(bpNewChartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'New Sys (mmHg)',
                data: [],
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0)',
                borderWidth: 2,
                tension: 0.4
            },
            {
                label: 'New Dia (mmHg)',
                data: [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0)',
                borderWidth: 2,
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { 
                display: false,
                grid: { color: '#334155' }
            },
            y: { 
                beginAtZero: false,
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' }
            }
        },
        plugins: {
            legend: { labels: { color: '#e2e8f0' } }
        },
        animation: false
    }
});


let pollingInterval = null;

startBtn.onclick = async () => {
    const user = nameInput.value.trim();
    if(!user) {
        alert("Please enter your name first!");
        return;
    }

    setFeedback("Starting session...", "yellow");

    try {
        const res = await fetch("/start", { 
            method: "POST",
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_name: user })
        });
        const data = await res.json();
        console.log(data);
        
        setFeedback(`Session started for ${user}`, "green");

        // Start video feed
        videoFeed.src = "/video_feed";
        
        // Reset charts and session data on start
        resetCharts();
        sessionHistory = { hr: [], sys: [], dia: [], sys_new: [], dia_new: [] };
        document.getElementById("sessionSummary").style.display = "none";

        // Start polling
        startPolling();
        
    } catch (e) {
        console.error("Error starting:", e);
        setFeedback("Error starting session", "red");
    }
};

stopBtn.onclick = async () => {
    setFeedback("Stopping session...", "yellow");
    try {
        const res = await fetch("/stop", { method: "POST" });
        const data = await res.json();
        console.log(data);
        
        setFeedback("Session stopped", "red");

        // Stop video feed (remove src to stop requests)
        videoFeed.src = "";
        
        stopPolling();
        
        // Calculate and display session summary
        displaySessionSummary();
        
    } catch (e) {
        console.error("Error stopping:", e);
        setFeedback("Error stopping session", "red");
    }
}

function setFeedback(msg, color) {
    feedback.innerText = msg;
    feedback.style.color = color === "green" ? "#22c55e" : (color === "red" ? "#ef4444" : "#facc15");
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(updateMetrics, 1000); // 1s interval
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function updateMetrics() {
    try {
        const res = await fetch("/data");
        const data = await res.json();
        
        // Update DOM
        setText("hr", data.hr, 1);
        setText("rr", data.rr, 1);
        
        const sys = data.sys ? data.sys.toFixed(0) : "--";
        const dia = data.dia ? data.dia.toFixed(0) : "--";
        document.getElementById("bp").innerText = `${sys}/${dia}`;

        const sys_new = data.sys_new ? data.sys_new.toFixed(0) : "--";
        const dia_new = data.dia_new ? data.dia_new.toFixed(0) : "--";
        document.getElementById("bp_new").innerText = `${sys_new}/${dia_new}`;
        
        setText("sqi", data.sqi, 2);
        
        document.getElementById("stress_label").innerText = data.stress_label || "--";
        setText("calmness", data.calmness_score, 2);
        
        document.getElementById("debugBox").innerText = JSON.stringify(data, null, 2);

        // Collect session history for averages
        sessionHistory.hr.push(data.hr !== undefined && data.hr !== null ? data.hr : null);
        sessionHistory.sys.push(data.sys !== undefined && data.sys !== null ? data.sys : null);
        sessionHistory.dia.push(data.dia !== undefined && data.dia !== null ? data.dia : null);
        sessionHistory.sys_new.push(data.sys_new !== undefined && data.sys_new !== null ? data.sys_new : null);
        sessionHistory.dia_new.push(data.dia_new !== undefined && data.dia_new !== null ? data.dia_new : null);

        // Update Charts
        updateCharts(data);
        
    } catch (e) {
        console.log("Polling error (engine might be stopped):", e);
    }
}

function setText(id, val, fixed) {
    const el = document.getElementById(id);
    if (val === null || val === undefined || isNaN(val)) {
        el.innerText = "--";
    } else {
        el.innerText = val.toFixed(fixed);
    }
}

function updateCharts(data) {
    const timeLabel = new Date().toLocaleTimeString();

    // -- HR --
    if (hrChart.data.labels.length > MAX_POINTS) {
        hrChart.data.labels.shift();
        hrChart.data.datasets[0].data.shift();
    }
    hrChart.data.labels.push(timeLabel);
    // Don't push NaN to charts if possible, or push it as null to break the line
    const hrVal = (data.hr && !isNaN(data.hr)) ? data.hr : null;
    hrChart.data.datasets[0].data.push(hrVal);
    hrChart.update();

    // -- BP --
    if (bpChart.data.labels.length > MAX_POINTS) {
        bpChart.data.labels.shift();
        bpChart.data.datasets.forEach(bs => bs.data.shift());
    }
    bpChart.data.labels.push(timeLabel);
    
    const sysVal = (data.sys && !isNaN(data.sys)) ? data.sys : null;
    const diaVal = (data.dia && !isNaN(data.dia)) ? data.dia : null;
    
    bpChart.data.datasets[0].data.push(sysVal);
    bpChart.data.datasets[1].data.push(diaVal);
    bpChart.update();

    // -- BP NEW --
    if (bpNewChart.data.labels.length > MAX_POINTS) {
        bpNewChart.data.labels.shift();
        bpNewChart.data.datasets.forEach(bs => bs.data.shift());
    }
    bpNewChart.data.labels.push(timeLabel);
    
    const sysNewVal = (data.sys_new && !isNaN(data.sys_new)) ? data.sys_new : null;
    const diaNewVal = (data.dia_new && !isNaN(data.dia_new)) ? data.dia_new : null;
    
    bpNewChart.data.datasets[0].data.push(sysNewVal);
    bpNewChart.data.datasets[1].data.push(diaNewVal);
    bpNewChart.update();
}

function resetCharts() {
    hrChart.data.labels = [];
    hrChart.data.datasets[0].data = [];
    hrChart.update();
    
    bpChart.data.labels = [];
    bpChart.data.datasets.forEach(d => d.data = []);
    bpChart.update();

    bpNewChart.data.labels = [];
    bpNewChart.data.datasets.forEach(d => d.data = []);
    bpNewChart.update();
}

function displaySessionSummary() {
    // Calculate averages from session history
    const validHr = sessionHistory.hr.filter(x => x !== null && x !== undefined && !isNaN(x));
    const validSys = sessionHistory.sys.filter(x => x !== null && x !== undefined && !isNaN(x));
    const validDia = sessionHistory.dia.filter(x => x !== null && x !== undefined && !isNaN(x));
    const validSysNew = sessionHistory.sys_new.filter(x => x !== null && x !== undefined && !isNaN(x));
    const validDiaNew = sessionHistory.dia_new.filter(x => x !== null && x !== undefined && !isNaN(x));

    const avgHr = validHr.length ? (validHr.reduce((a, b) => a + b, 0) / validHr.length).toFixed(1) : "--";
    const avgSys = validSys.length ? (validSys.reduce((a, b) => a + b, 0) / validSys.length).toFixed(0) : "--";
    const avgDia = validDia.length ? (validDia.reduce((a, b) => a + b, 0) / validDia.length).toFixed(0) : "--";
    const avgSysNew = validSysNew.length ? (validSysNew.reduce((a, b) => a + b, 0) / validSysNew.length).toFixed(0) : "--";
    const avgDiaNew = validDiaNew.length ? (validDiaNew.reduce((a, b) => a + b, 0) / validDiaNew.length).toFixed(0) : "--";

    // Update summary display
    document.getElementById("avgHr").innerText = avgHr + " bpm";
    document.getElementById("avgBpOld").innerText = `${avgSys}/${avgDia} mmHg`;
    document.getElementById("avgBpNew").innerText = `${avgSysNew}/${avgDiaNew} mmHg`;
    
    // Show summary section
    document.getElementById("sessionSummary").style.display = "block";
    
    // Scroll to summary
    document.getElementById("sessionSummary").scrollIntoView({ behavior: 'smooth' });
}
