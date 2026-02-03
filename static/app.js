
const videoFeed = document.getElementById("videoFeed");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let pollingInterval = null;

startBtn.onclick = async () => {
    try {
        const res = await fetch("/start", { method: "POST" });
        const data = await res.json();
        console.log(data);
        
        // Start video feed
        videoFeed.src = "/video_feed";
        
        // Start polling
        startPolling();
        
    } catch (e) {
        console.error("Error starting:", e);
    }
};

stopBtn.onclick = async () => {
    try {
        const res = await fetch("/stop", { method: "POST" });
        const data = await res.json();
        console.log(data);
        
        // Stop video feed (remove src to stop requests)
        videoFeed.src = "";
        
        stopPolling();
        
    } catch (e) {
        console.error("Error stopping:", e);
    }
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
        
        setText("sqi", data.sqi, 2);
        
        document.getElementById("stress_label").innerText = data.stress_label || "--";
        setText("calmness", data.calmness_score, 2);
        
        document.getElementById("debugBox").innerText = JSON.stringify(data, null, 2);
        
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
