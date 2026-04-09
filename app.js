import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const HAND_WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const FINGER_NAMES = ["엄지", "검지", "중지", "약지", "소지"];

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const canvasContext = canvas.getContext("2d");
const cameraToggle = document.getElementById("camera-toggle");
const statusMessage = document.getElementById("status-message");
const totalFingers = document.getElementById("total-fingers");
const totalHands = document.getElementById("total-hands");
const fpsValue = document.getElementById("fps-value");
const handList = document.getElementById("hand-list");

const drawingUtils = new DrawingUtils(canvasContext);

let handLandmarker;
let stream;
let running = false;
let animationFrameId = 0;
let lastVideoTime = -1;
let fpsWindowStart = performance.now();
let fpsFrames = 0;

window.addEventListener("resize", () => {
  if (video.srcObject) {
    resizeCanvas();
  }
});

init().catch((error) => {
  console.error(error);
  setStatus(`초기화 실패: ${error.message}`);
});

cameraToggle.addEventListener("click", async () => {
  if (running) {
    stopCamera();
    return;
  }

  if (!handLandmarker) {
    setStatus("모델이 아직 준비되지 않았다.");
    return;
  }

  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    setStatus(`카메라 시작 실패: ${error.message}`);
  }
});

async function init() {
  cameraToggle.disabled = true;
  setStatus("MediaPipe 모델 로딩 중");

  const vision = await FilesetResolver.forVisionTasks(HAND_WASM_ROOT);

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 4,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.45,
  });

  cameraToggle.disabled = false;
  setStatus("준비 완료. 카메라 시작을 누르면 추적을 시작한다.");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("이 브라우저는 getUserMedia를 지원하지 않는다.");
  }

  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });

  video.srcObject = stream;

  await video.play();
  resizeCanvas();

  running = true;
  lastVideoTime = -1;
  fpsWindowStart = performance.now();
  fpsFrames = 0;

  cameraToggle.textContent = "카메라 중지";
  setStatus("카메라 스트림 활성화. 손을 화면 안에 넣어라.");

  detectFrame();
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(animationFrameId);
  animationFrameId = 0;

  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
    stream = undefined;
  }

  video.srcObject = null;
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);
  updateSummary([], []);
  renderHandList([]);

  cameraToggle.textContent = "카메라 시작";
  setStatus("카메라 중지됨");
}

function resizeCanvas() {
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;

  canvas.width = width;
  canvas.height = height;
}

function detectFrame() {
  if (!running || !handLandmarker) {
    return;
  }

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const result = handLandmarker.detectForVideo(video, now);
    drawResult(result);
    updateFps(now);
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

function drawResult(result) {
  const landmarks = result.landmarks ?? [];
  const handedness = result.handedness ?? [];
  const handSummaries = [];

  canvasContext.clearRect(0, 0, canvas.width, canvas.height);

  landmarks.forEach((handLandmarks, index) => {
    const handed = handedness[index]?.[0];
    const handedLabel = handed?.categoryName ?? `Hand ${index + 1}`;
    const handedScore = handed?.score ?? 0;
    const fingerState = getFingerState(handLandmarks);
    const bounds = getBounds(handLandmarks, canvas.width, canvas.height);

    canvasContext.lineWidth = 3;
    canvasContext.strokeStyle = "rgba(63, 224, 197, 0.92)";
    canvasContext.fillStyle = "rgba(63, 224, 197, 0.12)";
    roundRect(
      canvasContext,
      bounds.minX,
      bounds.minY,
      bounds.width,
      bounds.height,
      18,
    );
    canvasContext.fill();
    canvasContext.stroke();

    drawingUtils.drawConnectors(handLandmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "#ff9e4f",
      lineWidth: 4,
    });
    drawingUtils.drawLandmarks(handLandmarks, {
      color: "#3fe0c5",
      radius: 5,
      lineWidth: 2,
      fillColor: "#07111a",
    });

    drawLabel({
      x: bounds.minX,
      y: Math.max(12, bounds.minY - 42),
      title: `${handedLabel} · ${fingerState.total}`,
      detail: `${Math.round(handedScore * 100)}% confidence`,
    });

    handSummaries.push({
      label: handedLabel,
      score: handedScore,
      fingers: fingerState.open,
      total: fingerState.total,
    });
  });

  updateSummary(handSummaries, landmarks);
  renderHandList(handSummaries);
}

function updateSummary(handSummaries, landmarks) {
  const total = handSummaries.reduce((sum, hand) => sum + hand.total, 0);

  totalFingers.textContent = String(total);
  totalHands.textContent = String(landmarks.length);
}

function updateFps(now) {
  fpsFrames += 1;
  const elapsed = now - fpsWindowStart;

  if (elapsed >= 500) {
    fpsValue.textContent = String(Math.round((fpsFrames * 1000) / elapsed));
    fpsFrames = 0;
    fpsWindowStart = now;
  }
}

function renderHandList(handSummaries) {
  if (handSummaries.length === 0) {
    handList.innerHTML =
      '<p class="empty-state">현재 인식된 손이 없다. 손 전체가 프레임 안에 들어오도록 맞춰라.</p>';
    return;
  }

  handList.innerHTML = handSummaries
    .map(
      (hand, index) => `
        <article class="hand-card">
          <div class="hand-card-header">
            <h3 class="hand-name">${escapeHtml(hand.label)} #${index + 1}</h3>
            <span class="hand-score">${Math.round(hand.score * 100)}% · ${hand.total} fingers</span>
          </div>
          <div class="finger-grid">
            ${hand.fingers
              .map(
                (active, fingerIndex) => `
                  <span class="finger-pill${active ? " active" : ""}">
                    ${FINGER_NAMES[fingerIndex]}
                  </span>
                `,
              )
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function getFingerState(landmarks) {
  const wrist = landmarks[0];
  const thumbAngle = angleBetween(landmarks[1], landmarks[2], landmarks[4]);
  const thumbReach =
    distance(landmarks[4], landmarks[5]) > distance(landmarks[3], landmarks[5]) * 1.1;

  const fingers = [
    thumbAngle > 143 &&
      thumbReach &&
      distance(landmarks[4], wrist) > distance(landmarks[2], wrist) * 0.9,
    isFingerExtended(landmarks, 5, 6, 8),
    isFingerExtended(landmarks, 9, 10, 12),
    isFingerExtended(landmarks, 13, 14, 16),
    isFingerExtended(landmarks, 17, 18, 20),
  ];

  return {
    open: fingers,
    total: fingers.filter(Boolean).length,
  };
}

function isFingerExtended(landmarks, mcpIndex, pipIndex, tipIndex) {
  const wrist = landmarks[0];
  const mcp = landmarks[mcpIndex];
  const pip = landmarks[pipIndex];
  const tip = landmarks[tipIndex];
  const bendAngle = angleBetween(mcp, pip, tip);

  return (
    bendAngle > 158 &&
    distance(tip, wrist) > distance(pip, wrist) * 1.08 &&
    distance(tip, mcp) > distance(pip, mcp) * 1.35
  );
}

function getBounds(landmarks, width, height) {
  const xs = landmarks.map((landmark) => landmark.x * width);
  const ys = landmarks.map((landmark) => landmark.y * height);
  const padding = 22;
  const minX = clamp(Math.min(...xs) - padding, 0, width);
  const minY = clamp(Math.min(...ys) - padding, 0, height);
  const maxX = clamp(Math.max(...xs) + padding, 0, width);
  const maxY = clamp(Math.max(...ys) + padding, 0, height);

  return {
    minX,
    minY,
    width: Math.max(0, maxX - minX),
    height: Math.max(0, maxY - minY),
  };
}

function drawLabel({ x, y, title, detail }) {
  canvasContext.save();
  canvasContext.font = "600 18px 'Space Grotesk'";
  const titleWidth = canvasContext.measureText(title).width;
  canvasContext.font = "500 12px 'IBM Plex Mono'";
  const detailWidth = canvasContext.measureText(detail).width;
  const width = Math.max(titleWidth, detailWidth) + 24;
  const height = 44;

  canvasContext.fillStyle = "rgba(7, 17, 26, 0.9)";
  canvasContext.strokeStyle = "rgba(255, 158, 79, 0.9)";
  roundRect(canvasContext, x, y, width, height, 14);
  canvasContext.fill();
  canvasContext.stroke();

  canvasContext.fillStyle = "#eff7fb";
  canvasContext.font = "600 18px 'Space Grotesk'";
  canvasContext.fillText(title, x + 12, y + 18);
  canvasContext.fillStyle = "#94aebe";
  canvasContext.font = "500 12px 'IBM Plex Mono'";
  canvasContext.fillText(detail, x + 12, y + 34);
  canvasContext.restore();
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function angleBetween(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const abMagnitude = Math.hypot(ab.x, ab.y, ab.z);
  const cbMagnitude = Math.hypot(cb.x, cb.y, cb.z);

  if (!abMagnitude || !cbMagnitude) {
    return 0;
  }

  const cosine = clamp(dot / (abMagnitude * cbMagnitude), -1, 1);
  return (Math.acos(cosine) * 180) / Math.PI;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function setStatus(message) {
  statusMessage.textContent = message;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
