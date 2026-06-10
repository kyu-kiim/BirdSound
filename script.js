// Safari / 모바일 브라우저에서는 사용자 제스처 없이 자동 카메라 요청이 차단될 수 있음.
// 해당 환경에서 작동하지 않을 경우 시작 버튼을 추가하는 방식으로 전환 필요.

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video         = document.getElementById("webcam");
const canvas        = document.getElementById("overlay");
const ctx           = canvas.getContext("2d");
const permissionMsg = document.getElementById("permissionMsg");

// ── MediaPipe ──────────────────────────────────────────────
let faceLandmarker = null;
let smoothX = 0;
let smoothY = 0;
let isFirstFrame = true;

// ── Audio ──────────────────────────────────────────────────
let audioStarted = false;
let dwellSoundIndex = -1;
let dwellStartTime  = 0;
const DWELL_DURATION = 4000;

const SOUND_DEFS = [
  { hour: 1,  name: "Chinese Blackbird",       file: "final_1_Chinese Blackbird.wav" },
  { hour: 3,  name: "Manchurian Bush Warbler", file: "finalize_3_Manchurian Bush Warbler .wav" },
  { hour: 5,  name: "Rufous-tailed Robin",     file: "finalize_5_Rufous-tailed Robin.wav" },
  { hour: 6,  name: "Marsh Tit",               file: "final_6_Marsh Tit.wav" },
  { hour: 9,  name: "Grey-headed Woodpecker",  file: "final9_Grey-headed Woodpecker .wav" },
  { hour: 11, name: "Japanese Wood Pigeon",    file: "final11_Japanese Wodd Pigeon.wav" },
  { hour: 12, name: "Yellow-billed Grosbeak",  file: "final_12_Yellow-billed Grosbeak.wav" },
];
let sounds = [];

// ── Particle system ────────────────────────────────────────
const PARTICLE_COUNT   = 22;
const MAX_SCATTER_R    = 28;   // px, 흩어진 상태의 최대 반경
const CONVERGE_DELAY   = 2200; // ms, 완전히 뭉치는 데 걸리는 시간
const SCATTER_DECAY    = 2.5;  // 이동 시 흩어지는 속도 배수
const VELOCITY_THRESH  = 1.8;  // px/frame, 이 값 이하면 정지로 간주

let particles = [];
let convergeFactor = 0;  // 0 = 완전 흩어짐, 1 = 완전 뭉침
let gazeConvergeTime = 0; // ms 누적
let prevGazeX = 0;
let prevGazeY = 0;
let lastFrameTime = 0;

function initParticles() {
  particles = Array.from({ length: PARTICLE_COUNT }, () => {
    const angle = Math.random() * Math.PI * 2;
    const r     = 0.35 + Math.random() * 0.65;
    return {
      angle,
      dAngle:  (Math.random() - 0.5) * 0.055,   // 회전 속도 (±)
      rPhase:  Math.random() * Math.PI * 2,       // 반경 미세 진동 위상
      rSpeed:  Math.random() * 0.025 + 0.008,    // 반경 진동 속도
      r,                                          // 정규화 반경 (0~1)
      ox: Math.cos(angle) * r * MAX_SCATTER_R,   // 초기 오프셋 (흩어진 상태)
      oy: Math.sin(angle) * r * MAX_SCATTER_R,
      size:    1.2 + Math.random() * 2.0,
      opacity: 0.35 + Math.random() * 0.5,
    };
  });
}

// 파티클 + 수렴 점 그리기
// faceAlpha: 얼굴 인식 시 1.0, 미인식 시 0.15
function drawGazeVisual(cx, cy, faceAlpha) {
  // ── 파티클 (흩어진 상태 ~ 수렴 직전) ──
  const particleGlobalAlpha = 1 - Math.pow(convergeFactor, 1.5);

  if (particleGlobalAlpha > 0.01) {
    particles.forEach(p => {
      p.angle  += p.dAngle;
      p.rPhase += p.rSpeed;

      // 반경에 미세한 유기적 진동 추가
      const radiusMod = 0.85 + 0.15 * Math.sin(p.rPhase);
      const targetR   = p.r * radiusMod * MAX_SCATTER_R * (1 - convergeFactor);

      const tx = Math.cos(p.angle) * targetR;
      const ty = Math.sin(p.angle) * targetR;

      // 목표 위치로 부드럽게 이동
      p.ox += (tx - p.ox) * 0.09;
      p.oy += (ty - p.oy) * 0.09;

      ctx.save();
      ctx.globalAlpha = faceAlpha * p.opacity * particleGlobalAlpha;
      ctx.beginPath();
      ctx.arc(cx + p.ox, cy + p.oy, p.size, 0, Math.PI * 2);
      ctx.fillStyle = "white";
      ctx.fill();
      ctx.restore();
    });
  }

  // ── 수렴 점 (뭉치는 상태) ──
  const dotAlpha = Math.pow(convergeFactor, 1.5);
  if (dotAlpha > 0.01) {
    // 부드러운 외곽 glow
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 22);
    glow.addColorStop(0, `rgba(255,255,255,${dotAlpha * 0.25})`);
    glow.addColorStop(1,  "rgba(255,255,255,0)");
    ctx.save();
    ctx.globalAlpha = faceAlpha;
    ctx.beginPath();
    ctx.arc(cx, cy, 22, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();

    // 중심 흰색 점 (반경 2 ~ 9px)
    const dotR = 2 + 7 * convergeFactor;
    ctx.globalAlpha = faceAlpha * dotAlpha;
    ctx.beginPath();
    ctx.arc(cx, cy, dotR, 0, Math.PI * 2);
    ctx.fillStyle = "white";
    ctx.fill();
    ctx.restore();
  }
}

// ── Sound positions ────────────────────────────────────────
function edgePosition(hour, cx, cy, w, h, padding = 0.90) {
  const angle = (hour / 12) * 2 * Math.PI;
  const dx = Math.sin(angle);
  const dy = -Math.cos(angle);
  let t = Infinity;
  if (dx > 0) t = Math.min(t, (w - cx) / dx);
  if (dx < 0) t = Math.min(t, -cx / dx);
  if (dy > 0) t = Math.min(t, (h - cy) / dy);
  if (dy < 0) t = Math.min(t, -cy / dy);
  return { x: cx + t * dx * padding, y: cy + t * dy * padding };
}

function computeSoundPositions() {
  const cx = canvas.width / 2, cy = canvas.height / 2;
  sounds.forEach(s => {
    const p = edgePosition(s.hour, cx, cy, canvas.width, canvas.height);
    s.x = p.x; s.y = p.y;
  });
}

// 테스트용: 소리 위치 점 + 이름
function drawSoundPositions() {
  const cx = canvas.width / 2, cy = canvas.height / 2;
  sounds.forEach(s => {
    const dx = s.x - cx, dy = s.y - cy;
    const len = Math.sqrt(dx * dx + dy * dy);
    const nx = dx / len, ny = dy / len;

    ctx.beginPath();
    ctx.arc(s.x, s.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fill();

    const lx = s.x + nx * 18, ly = s.y + ny * 18;
    ctx.font         = "11px 'Segoe UI', sans-serif";
    ctx.fillStyle    = "rgba(255,255,255,0.55)";
    ctx.textBaseline = Math.abs(nx) >= Math.abs(ny) ? "middle" : (ny > 0 ? "top" : "bottom");
    ctx.textAlign    = Math.abs(nx) >= Math.abs(ny) ? (nx > 0 ? "left" : "right") : "center";
    ctx.fillText(s.name, lx, ly);
  });
}

// ── Audio init ─────────────────────────────────────────────
function initAudio() {
  sounds = SOUND_DEFS.map(def => {
    const audio = new Audio(`sounds/${encodeURIComponent(def.file)}`);
    audio.loop = true;
    audio.volume = 0.5;
    return { audio, hour: def.hour, name: def.name, x: 0, y: 0 };
  });
  computeSoundPositions();
}

function startAudio() {
  if (audioStarted) return;
  audioStarted = true;
  permissionMsg.textContent = "";
  sounds.forEach(s => s.audio.play().catch(err => console.error(`오디오 재생 실패 [${s.hour}시]:`, err)));
}

function setupAudioUnlock() {
  const unlock = () => {
    startAudio();
    window.removeEventListener("click",    unlock);
    window.removeEventListener("touchend", unlock);
  };
  window.addEventListener("click",    unlock);
  window.addEventListener("touchend", unlock);
}

function updateVolumes(gazeX, gazeY) {
  const now = performance.now();
  const cx  = canvas.width / 2, cy = canvas.height / 2;

  const distFromCenter = Math.sqrt((gazeX - cx) ** 2 + (gazeY - cy) ** 2);
  const maxDist        = Math.sqrt(cx ** 2 + cy ** 2);
  const focusStrength  = Math.min(distFromCenter / maxDist, 1);

  let minDist = Infinity, closestIdx = 0;
  sounds.forEach((s, i) => {
    const d = Math.sqrt((gazeX - s.x) ** 2 + (gazeY - s.y) ** 2);
    if (d < minDist) { minDist = d; closestIdx = i; }
  });

  if (focusStrength < 0.25) {
    dwellSoundIndex = -1;
    sounds.forEach(s => { s.audio.volume += (0.5 - s.audio.volume) * 0.06; });
    return;
  }

  if (closestIdx !== dwellSoundIndex) {
    dwellSoundIndex = closestIdx;
    dwellStartTime  = now;
  }

  const dwellProgress = Math.min((now - dwellStartTime) / DWELL_DURATION, 1);
  const targetFocused = 0.70 + 0.20 * dwellProgress;

  sounds.forEach((s, i) => {
    const target = (i === closestIdx) ? targetFocused : 0.05;
    s.audio.volume += (target - s.audio.volume) * 0.06;
    s.audio.volume  = Math.max(0, Math.min(1, s.audio.volume));
  });
}

// ── Canvas resize ──────────────────────────────────────────
function resizeCanvas() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener("resize", () => { resizeCanvas(); computeSoundPositions(); });
resizeCanvas();

// ── MediaPipe init ─────────────────────────────────────────
async function initModel() {
  console.log("모델 준비 중...");
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  console.log("모델 준비 완료");
}

async function startCamera() {
  console.log("카메라 권한 요청 중...");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  video.srcObject = stream;
  video.play();
  video.addEventListener("loadeddata", () => {
    console.log("얼굴 방향 추적 중...");
    lastFrameTime = performance.now();
    detectLoop();
  });
}

// ── Main loop ──────────────────────────────────────────────
function detectLoop() {
  if (!faceLandmarker) { requestAnimationFrame(detectLoop); return; }

  const now = performance.now();
  const dt  = now - lastFrameTime;
  lastFrameTime = now;

  const results = faceLandmarker.detectForVideo(video, now);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawSoundPositions();

  if (results.faceLandmarks && results.faceLandmarks.length > 0) {
    const lm = results.faceLandmarks[0];

    const noseTip     = lm[1];
    const leftEye     = lm[33];
    const rightEye    = lm[263];
    const betweenEyes = lm[168];
    const mouthCenter = lm[13];

    const faceCenterX = (leftEye.x + rightEye.x) / 2;
    const eyeDistance = Math.abs(rightEye.x - leftEye.x);
    let yaw = -((noseTip.x - faceCenterX) / eyeDistance);

    const faceCenterY = (betweenEyes.y + mouthCenter.y) / 2;
    const faceHeight  = Math.abs(mouthCenter.y - betweenEyes.y);
    let pitch = (noseTip.y - faceCenterY) / faceHeight;
    // pitch = -pitch; // 위아래 반전 필요 시 주석 해제

    let pointX = canvas.width  / 2 + yaw   * canvas.width  * 1.5;
    let pointY = canvas.height / 2 + pitch * canvas.height * 1.5;
    pointX = Math.max(0, Math.min(canvas.width,  pointX));
    pointY = Math.max(0, Math.min(canvas.height, pointY));

    if (isFirstFrame) {
      smoothX = pointX; smoothY = pointY;
      prevGazeX = pointX; prevGazeY = pointY;
      isFirstFrame = false;
    } else {
      smoothX += (pointX - smoothX) * 0.12;
      smoothY += (pointY - smoothY) * 0.12;
    }

    // ── convergeFactor 업데이트 ──
    const vel = Math.sqrt((smoothX - prevGazeX) ** 2 + (smoothY - prevGazeY) ** 2);
    prevGazeX = smoothX;
    prevGazeY = smoothY;

    if (vel < VELOCITY_THRESH) {
      gazeConvergeTime = Math.min(gazeConvergeTime + dt, CONVERGE_DELAY);
    } else {
      gazeConvergeTime = Math.max(gazeConvergeTime - dt * SCATTER_DECAY, 0);
    }
    convergeFactor = gazeConvergeTime / CONVERGE_DELAY;

    updateVolumes(smoothX, smoothY);
    drawGazeVisual(smoothX, smoothY, 1.0);
  } else {
    console.log("얼굴을 찾는 중...");
    // 얼굴 미인식: 파티클 흩어진 채 희미하게 유지
    gazeConvergeTime = Math.max(gazeConvergeTime - dt * SCATTER_DECAY, 0);
    convergeFactor   = gazeConvergeTime / CONVERGE_DELAY;
    drawGazeVisual(smoothX, smoothY, 0.15);
  }

  requestAnimationFrame(detectLoop);
}

// ── App entry ──────────────────────────────────────────────
async function startApp() {
  try {
    initParticles();
    initAudio();
    setupAudioUnlock();
    permissionMsg.textContent = "Click anywhere to start audio";
    await initModel();
    await startCamera();
  } catch (err) {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      permissionMsg.textContent = "Camera permission required";
    } else {
      console.error("앱 초기화 실패:", err);
    }
  }
}

window.addEventListener("DOMContentLoaded", startApp);
