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

// ── Calibration ────────────────────────────────────────────
const CALIB_FRAMES  = 90;
let neutralYaw      = 0;
let neutralPitch    = 0;
let neutralIrisX    = 0;
let neutralIrisY    = 0;
let isCalibrated    = false;
let calibYawSum     = 0;
let calibPitchSum   = 0;
let calibIrisXSum   = 0;
let calibIrisYSum   = 0;
let calibFrameCount = 0;

function resetCalibration() {
  isCalibrated    = false;
  calibYawSum     = 0;
  calibPitchSum   = 0;
  calibIrisXSum   = 0;
  calibIrisYSum   = 0;
  calibFrameCount = 0;
  console.log("캘리브레이션 재시작");
}

document.addEventListener("keydown", e => {
  if (e.key === "c" || e.key === "C") resetCalibration();
});

// ── Audio (Web Audio API) ──────────────────────────────────
let audioStarted    = false;
let dwellSoundIndex = -1;
let dwellStartTime  = 0;
const DWELL_DURATION = 4000;

let audioCtx   = null;
let masterGain = null;

// masterGain을 통한 전체 페이드 (Web Audio 스케줄링 사용)
// timeConst 0.15 ≒ 0.5초 fade
function fadeGlobalTo(target) {
  if (!masterGain) return;
  const now = audioCtx.currentTime;
  masterGain.gain.cancelScheduledValues(now);
  masterGain.gain.setTargetAtTime(target, now, 0.15);
}

// 탭 전환 시 페이드 아웃/인 + AudioContext suspend/resume
document.addEventListener("visibilitychange", () => {
  if (!audioCtx) return;
  if (document.hidden) {
    fadeGlobalTo(0);
    setTimeout(() => {
      if (audioCtx.state === "running") audioCtx.suspend();
    }, 600);
  } else {
    audioCtx.resume().then(() => fadeGlobalTo(1));
  }
});

// 각 채널 평균 RMS 계산
function calcRMS(buffer) {
  let sumSq = 0, total = 0;
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) sumSq += data[i] * data[i];
    total += data.length;
  }
  return Math.sqrt(sumSq / total);
}

// ── Sound definitions ──────────────────────────────────────
// 하단 5개 (고개 약 5° 이내): left → 왼쪽, right → 오른쪽, 방향 없음 → 중앙
// 상단 3개 (고개 약 20° 이상): 동일 원칙
const SOUND_DEFS = [
  // 하단 줄 (yFrac 0.30) — 좌 → 우 순서, 간격 약 20%씩 균등
  { name: "Grey-headed Woodpecker",  file: "left_Grey-headed Woodpecker .wav",           xFrac: 0.10, yFrac: 0.57 },
  { name: "Japanese Bush Warbler",   file: "left_japanese bush warbler.wav",              xFrac: 0.28, yFrac: 0.57 },
  { name: "Marsh Tit",               file: "final_6_Marsh Tit.wav",                      xFrac: 0.50, yFrac: 0.57 },
  { name: "Manchurian Bush Warbler", file: "right_3_Manchurian Bush Warbler .wav",       xFrac: 0.72, yFrac: 0.57 },
  { name: "Chinese Blackbird",       file: "right_1_Chinese Blackbird.wav",              xFrac: 0.90, yFrac: 0.57 },
  // 상단 줄 — 좌 → 우 순서
  { name: "Japanese Wood Pigeon",    file: "left_Japanese Wodd Pigeon.wav",              xFrac: 0.20, yFrac: 0.44 },
  { name: "Yellow-billed Grosbeak",  file: "final_12_Yellow-billed Grosbeak.wav",        xFrac: 0.50, yFrac: 0.44 },
  { name: "Rufous-tailed Robin",     file: "right_Rufous-tailed Robin.wav",              xFrac: 0.80, yFrac: 0.44 },
];

let sounds = [];

function computeSoundPositions() {
  sounds.forEach(s => {
    s.x = s.xFrac * canvas.width;
    s.y = s.yFrac * canvas.height;
  });
}

// 오디오 객체 초기화 (Web Audio 노드는 startAudio에서 생성)
function initAudio() {
  sounds = SOUND_DEFS.map(def => ({
    name:      def.name,
    file:      def.file,
    xFrac:     def.xFrac,
    yFrac:     def.yFrac,
    x: 0, y: 0,
    smoothVol: 0.3,
    normGain:  null,   // GainNode: 정규화 고정값
    volGain:   null,   // GainNode: 동적 볼륨 제어
    source:    null,   // AudioBufferSourceNode
  }));
  computeSoundPositions();
}

async function startAudio() {
  if (audioStarted) return;
  audioStarted = true;
  permissionMsg.textContent = "";

  // 사용자 인터랙션 이후에 AudioContext 생성
  audioCtx   = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0;  // 페이드인 전까지 무음
  masterGain.connect(audioCtx.destination);

  // ── 1. 모든 파일 병렬 fetch + decode ──────────────────────
  const buffers = await Promise.all(
    sounds.map(async s => {
      try {
        const res  = await fetch(`sounds/${encodeURIComponent(s.file)}`);
        const data = await res.arrayBuffer();
        return await audioCtx.decodeAudioData(data);
      } catch (err) {
        console.error(`오디오 로드 실패 [${s.name}]:`, err);
        return null;
      }
    })
  );

  // ── 2. RMS 계산 및 정규화 타겟 결정 (전체 평균) ──────────
  const rmsValues = buffers.map(buf => buf ? calcRMS(buf) : 0);
  const validRMS  = rmsValues.filter(r => r > 0);
  const targetRMS = validRMS.length > 0
    ? validRMS.reduce((a, b) => a + b, 0) / validRMS.length
    : 0.1;

  console.log(
    "RMS 분석 결과:",
    rmsValues.map((r, i) => `${sounds[i].name}: ${r.toFixed(4)}`).join(" | ")
  );
  console.log(`타겟 RMS: ${targetRMS.toFixed(4)}`);

  // ── 3. 각 소리 노드 생성 및 연결 ──────────────────────────
  // 그래프: source → normGain → volGain → masterGain → destination
  sounds.forEach((s, i) => {
    const buf = buffers[i];
    if (!buf) return;

    // 정규화 gain (고정): targetRMS / fileRMS, 0.2~5.0 사이로 클램프
    const normGain = audioCtx.createGain();
    normGain.gain.value = rmsValues[i] > 0
      ? Math.min(Math.max(targetRMS / rmsValues[i], 0.2), 5.0)
      : 1.0;
    console.log(`  ${s.name} → normGain: ${normGain.gain.value.toFixed(3)}`);

    // 볼륨 제어 gain (동적, 0~1)
    const volGain = audioCtx.createGain();
    volGain.gain.value = 0.3;

    // 루프 소스
    const source = audioCtx.createBufferSource();
    source.buffer = buf;
    source.loop   = true;

    source.connect(normGain);
    normGain.connect(volGain);
    volGain.connect(masterGain);
    source.start();

    s.normGain = normGain;
    s.volGain  = volGain;
    s.source   = source;
  });

  // ── 4. 0.5초 페이드인 ─────────────────────────────────────
  masterGain.gain.setTargetAtTime(1, audioCtx.currentTime, 0.15);
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

// Vol 전환에 쓸 시정수(time constant):
//   attack (소리 커질 때): 0.08s — 빠른 반응
//   release (소리 작아질 때): 0.18s — 부드러운 감쇠, 순간 공백 방지
const TC_ATTACK  = 0.08;
const TC_RELEASE = 0.18;
const BASE_VOL   = 0.3;

function setVolSmooth(s, target) {
  if (!s.volGain || !audioCtx) return;
  const now     = audioCtx.currentTime;
  const clamped = Math.max(0, Math.min(1, target));
  const rising  = clamped > s.smoothVol;
  s.volGain.gain.cancelAndHoldAtTime(now);
  s.volGain.gain.setTargetAtTime(clamped, now, rising ? TC_ATTACK : TC_RELEASE);
  s.smoothVol = clamped;
}

function updateVolumes(gazeX, gazeY) {
  const now = performance.now();

  const threshold    = canvas.height * 0.62;
  const distAbove    = Math.max(0, threshold - gazeY);
  const focusStrength = Math.min(distAbove / threshold, 1);

  let minDist = Infinity, closestIdx = 0;
  sounds.forEach((s, i) => {
    const d = Math.sqrt((gazeX - s.x) ** 2 + (gazeY - s.y) ** 2);
    if (d < minDist) { minDist = d; closestIdx = i; }
  });

  if (focusStrength < 0.08) {
    dwellSoundIndex = -1;
    sounds.forEach(s => setVolSmooth(s, BASE_VOL));
    return;
  }

  if (closestIdx !== dwellSoundIndex) {
    dwellSoundIndex = closestIdx;
    dwellStartTime  = now;
  }

  const dwellProgress = Math.min((now - dwellStartTime) / DWELL_DURATION, 1);
  const targetFocused = 0.70 + 0.20 * dwellProgress;

  sounds.forEach((s, i) => {
    setVolSmooth(s, (i === closestIdx) ? targetFocused : 0.02);
  });
}

// ── Particle system ────────────────────────────────────────
const PARTICLE_COUNT  = 22;
const MAX_SCATTER_R   = 16;
const CONVERGE_DELAY  = 2200;
const SCATTER_DECAY   = 2.5;
const VELOCITY_THRESH = 1.8;

let particles        = [];
let convergeFactor   = 0;
let gazeConvergeTime = 0;
let prevGazeX        = 0;
let prevGazeY        = 0;
let lastFrameTime    = 0;

function initParticles() {
  particles = Array.from({ length: PARTICLE_COUNT }, () => {
    const angle = Math.random() * Math.PI * 2;
    const r     = 0.35 + Math.random() * 0.65;
    return {
      angle,
      dAngle:  (Math.random() - 0.5) * 0.055,
      rPhase:  Math.random() * Math.PI * 2,
      rSpeed:  Math.random() * 0.025 + 0.008,
      r,
      ox: Math.cos(angle) * r * MAX_SCATTER_R,
      oy: Math.sin(angle) * r * MAX_SCATTER_R,
      size:    0.7 + Math.random() * 1.0,
      opacity: 0.35 + Math.random() * 0.5,
    };
  });
}

function drawGazeVisual(cx, cy, faceAlpha) {
  const particleAlpha = 1 - Math.pow(convergeFactor, 1.5);

  if (particleAlpha > 0.01) {
    particles.forEach(p => {
      p.angle  += p.dAngle;
      p.rPhase += p.rSpeed;
      const radiusMod = 0.85 + 0.15 * Math.sin(p.rPhase);
      const targetR   = p.r * radiusMod * MAX_SCATTER_R * (1 - convergeFactor);
      const tx = Math.cos(p.angle) * targetR;
      const ty = Math.sin(p.angle) * targetR;
      p.ox += (tx - p.ox) * 0.09;
      p.oy += (ty - p.oy) * 0.09;

      ctx.save();
      ctx.globalAlpha = faceAlpha * p.opacity * particleAlpha;
      ctx.beginPath();
      ctx.arc(cx + p.ox, cy + p.oy, p.size, 0, Math.PI * 2);
      ctx.fillStyle = "white";
      ctx.fill();
      ctx.restore();
    });
  }

  const dotAlpha = Math.pow(convergeFactor, 1.5);
  if (dotAlpha > 0.01) {
    const dotR = 1.5 + 2.5 * convergeFactor;
    ctx.save();
    ctx.globalAlpha = faceAlpha * dotAlpha;
    ctx.beginPath();
    ctx.arc(cx, cy, dotR, 0, Math.PI * 2);
    ctx.fillStyle = "white";
    ctx.fill();
    ctx.restore();
  }
}

// 캘리브레이션 십자핀 (원 없는 공백 십자)
function drawCalibCrosshair(x, y) {
  const ARM = 10, GAP = 4;
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.60)";
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(x - ARM, y); ctx.lineTo(x - GAP, y);
  ctx.moveTo(x + GAP, y); ctx.lineTo(x + ARM, y);
  ctx.moveTo(x, y - ARM); ctx.lineTo(x, y - GAP);
  ctx.moveTo(x, y + GAP); ctx.lineTo(x, y + ARM);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 1.5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.60)";
  ctx.fill();
  ctx.restore();
}

// 시선이 소리 위치 위에 있을 때만 새 이름 표시
function drawBirdName(gazeX, gazeY) {
  const RADIUS = canvas.height * 0.09;
  sounds.forEach(s => {
    const d = Math.sqrt((gazeX - s.x) ** 2 + (gazeY - s.y) ** 2);
    if (d >= RADIUS) return;
    const alpha = Math.pow(1 - d / RADIUS, 2);
    ctx.save();
    ctx.globalAlpha      = alpha;
    ctx.font             = "13px 'Segoe UI', sans-serif";
    ctx.fillStyle        = "white";
    ctx.textAlign        = "center";
    ctx.textBaseline     = "middle";
    ctx.letterSpacing    = "0.06em";
    ctx.fillText(s.name, s.x, s.y);
    ctx.restore();
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
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 }, audio: false,
  });
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
    pitch = -pitch;

    // 홍채 보정값 (yaw 부호 관례에 맞춰 X 반전)
    let irisRawX = 0, irisRawY = 0;
    if (lm.length > 477) {
      const liC = lm[468], riC = lm[473];
      const liIn = lm[133], riIn = lm[362];
      const leW = Math.abs(leftEye.x  - liIn.x) || 0.01;
      const reW = Math.abs(rightEye.x - riIn.x) || 0.01;
      const leH = Math.abs(lm[145].y  - lm[159].y) || 0.01;
      const reH = Math.abs(lm[374].y  - lm[386].y) || 0.01;
      irisRawX = -(
        (liC.x - (leftEye.x  + liIn.x) / 2) / leW +
        (riC.x - (rightEye.x + riIn.x) / 2) / reW
      ) / 2;
      irisRawY = (
        (liC.y - (lm[159].y + lm[145].y) / 2) / leH +
        (riC.y - (lm[386].y + lm[374].y) / 2) / reH
      ) / 2;
    }

    // ── 캘리브레이션 ──────────────────────────────────────
    if (!isCalibrated) {
      calibYawSum   += yaw;
      calibPitchSum += pitch;
      calibIrisXSum += irisRawX;
      calibIrisYSum += irisRawY;
      calibFrameCount++;

      if (calibFrameCount >= CALIB_FRAMES) {
        neutralYaw   = calibYawSum   / calibFrameCount;
        neutralPitch = calibPitchSum / calibFrameCount;
        neutralIrisX = calibIrisXSum / calibFrameCount;
        neutralIrisY = calibIrisYSum / calibFrameCount;
        isCalibrated = true;
        console.log(`캘리브레이션 완료 — yaw: ${neutralYaw.toFixed(4)}, pitch: ${neutralPitch.toFixed(4)}`);
      } else {
        drawCalibCrosshair(canvas.width * 0.5, canvas.height * 0.70);
        requestAnimationFrame(detectLoop);
        return;
      }
    }

    // 개인 정면 보정값 차감 후 증폭
    const adjYaw   = yaw       - neutralYaw;
    const adjPitch = pitch     - neutralPitch;
    const adjIrisX = irisRawX  - neutralIrisX;
    const adjIrisY = irisRawY  - neutralIrisY;

    // 기준점: 화면 하단 70%
    const anchorX = canvas.width  * 0.50;
    const anchorY = canvas.height * 0.70;

    let pointX = anchorX - adjYaw   * canvas.width  * 1.5  - adjIrisX * canvas.width  * 0.15;
    let pointY = anchorY - adjPitch * canvas.height * 2.8  + adjIrisY * canvas.height * 0.12;
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
    drawBirdName(smoothX, smoothY);
  } else {
    gazeConvergeTime = Math.max(gazeConvergeTime - dt * SCATTER_DECAY, 0);
    convergeFactor   = gazeConvergeTime / CONVERGE_DELAY;
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
