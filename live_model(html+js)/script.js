// ===============================
//  CONFIG
// ===============================
const MODEL_INPUT_SIZE = 512; // Your model expects 512x512 inputs
const CLASS_NAMES = {
    0: "UNCC HEADGEAR",
    1: "UNCC TORSO",
    2: "UNCC-LOGO"
};

// ===============================
//  SETUP
// ===============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const fpsDisplay = document.getElementById("fpsDisplay");

let session;
let lastTime = performance.now();

// ===============================
//  LOAD MODEL
// ===============================
async function loadModel() {
    console.log("Loading model...");
    session = await ort.InferenceSession.create("CMM-Yolo11.onnx");
    console.log("Model loaded!");
}

// ===============================
//  START CAMERA
// ===============================
async function startCamera() {
    try {
        console.log("Requesting camera...");
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });

        console.log("Camera stream obtained!");
        video.srcObject = stream;

        // Important: force playback (Chrome requires this)
        await video.play();

        video.onloadedmetadata = () => {
            console.log("Video metadata loaded:", video.videoWidth, video.videoHeight);

            document.getElementById("video-container").style.display = "block";

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            detectFrame();
        };
    } catch (err) {
        console.error("Camera error:", err);
        alert("Camera error: " + err.message);
    }
}

// ===============================
//  MAIN DETECTION LOOP
// ===============================
async function detectFrame() {
    const now = performance.now();
    const fps = (1000 / (now - lastTime)).toFixed(1);
    lastTime = now;
    fpsDisplay.innerText = `FPS: ${fps}`;

    // Resize webcam image to 512x512 for the model
    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = MODEL_INPUT_SIZE;
    tmpCanvas.height = MODEL_INPUT_SIZE;
    const tmpCtx = tmpCanvas.getContext("2d");
    tmpCtx.drawImage(video, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

    // Read image pixels (RGBA)
    const imgData = tmpCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const data = imgData.data;

    // Build CHW format RGB array
    const input = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    let idx = 0;

    // R
    for (let i = 0; i < data.length; i += 4)
        input[idx++] = data[i] / 255;

    // G
    for (let i = 0; i < data.length; i += 4)
        input[idx++] = data[i + 1] / 255;

    // B
    for (let i = 0; i < data.length; i += 4)
        input[idx++] = data[i + 2] / 255;

    const tensor = new ort.Tensor("float32", input, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);

    // Model inference
    const output = await session.run({ images: tensor });
    const preds = output.output0.data;  // shape: [1, 7, 5376]

    drawDetections(preds);

    requestAnimationFrame(detectFrame);
}

// ===============================
//  DRAW DETECTIONS
// ===============================
function drawDetections(preds) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const NUM_BOXES = 5376;   // from your ONNX output
    const STRIDE = 7;         // each prediction = 7 values

    for (let i = 0; i < NUM_BOXES; i++) {
        const base = i * STRIDE;

        const x = preds[base + 0];   // center x
        const y = preds[base + 1];   // center y
        const w = preds[base + 2];   // width
        const h = preds[base + 3];   // height
        const obj = preds[base + 4]; // object confidence
        const cls = Math.round(preds[base + 5]); // class ID

        if (obj < 0.35) continue; // threshold

        const label = CLASS_NAMES[cls] || "Unknown";

        // Convert from 512-space to video resolution
        const left = (x - w / 2) * (canvas.width / MODEL_INPUT_SIZE);
        const top = (y - h / 2) * (canvas.height / MODEL_INPUT_SIZE);
        const bw = w * (canvas.width / MODEL_INPUT_SIZE);
        const bh = h * (canvas.height / MODEL_INPUT_SIZE);

        // Draw box
        ctx.strokeStyle = "#046A38"; // UNCC green
        ctx.lineWidth = 3;
        ctx.strokeRect(left, top, bw, bh);

        // Draw label
        ctx.fillStyle = "#FFD700";   // gold
        ctx.font = "20px Arial";
        ctx.fillText(`${label} (${(obj * 100).toFixed(1)}%)`, left, top - 6);
    }
}

// ===============================
//  START BUTTON CLICK
// ===============================
startBtn.onclick = async () => {
    startBtn.innerText = "Loading model...";
    startBtn.disabled = true;

    await loadModel();
    await startCamera();

    startBtn.style.display = "none";
};
