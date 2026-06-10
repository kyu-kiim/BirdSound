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
// 사람마다 다른 정면 각도를 보정하기 위해 처음 N 프레임의 평균을 neutral 값으로 저장
const CALIB_FRAMES  = 90;   // ~1.5초 (60fps 기준)
let neutralYaw      = 0;
let neutralPitch    = 0;
let isCalibrated    = false;
let calibYawSum     = 0;
let calibPitchSum   = 0;
let calibFrameCount = 0;

function resetCalibration() {
  isCalibrated    = false;
  calibYawSum     = 0;
  calibPitchSum   = 0;
  calibFrameCount = 0;
  console.log("캘리브레이션 재시작");
}

// 'C' 키로 언제든 재캘리브레이션 가능
document.addEventListener("keydown", e => {
  if (e.key === "c" || e.key === "C") resetCalibration();
});

// ── Audio ──────────────────────────────────────────────────
let audioStarted = false;
let dwellSoundIndex = -1;
let dwellStartTime  = 0;
const DWELL_DURATION = 4000;

// xFrac: 화면 너비 비율(0~1), yFrac: 화면 높이 비율(0~1, 위=작은값)
// ─ 하단 줄 (고개 약 5° 이내, 좌우 탐색 구간): 4 sounds ─────────────
// ─ 상단 줄 (고개 약 20° 이상, 위쪽 구간)   : 3 sounds ─────────────
// 소리가 9개가 되면 하단 5개 / 상단 4개로 완성
const SOUND_DEFS = [
  // 하단 줄 (yFrac ≈ 0.28)
  { name: "Chinese Blackbird",       file: "final_1_Chinese Blackbird.wav",              xFrac: 0.12, yFrac: 0.28 },
  { name: "Manchurian Bush Warbler", file: "finalize_3_Manchurian Bush Warbler .wav",    xFrac: 0.37, yFrac: 0.28 },
  { name: "Grey-headed Woodpecker",  file: "final9_Grey-headed Woodpecker .wav",         xFrac: 0.63, yFrac: 0.28 },
  { name: "Yellow-billed Grosbeak",  file: "final_12_Yellow-billed Grosbeak.wav",        xFrac: 0.88, yFrac: 0.28 },
  // 상단 줄 (yFrac ≈ 0.10)
  { name: "Rufous-tailed Robin",     file: "finalize_5_Rufous-tailed Robin.wav",         xFrac: 0.20, yFrac: 0.10 },
  { name: "Marsh Tit",               file: "final_6_Marsh Tit.wav",                      xFrac: 0.50, yFrac: 0.10 },
  { name: "Japanese Wood Pigeon",    file: "final11_Japanese Wodd Pigeon.wav",           xFrac: 0.80, yFrac: 0.10 },
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
function computeSoundPositions() {
  sounds.forEach(s => {
    s.x = s.xFrac * canvas.width;
    s.y = s.yFrac * canvas.height;
  });
}

// 캘리브레이션 중 표시되는 십자 조준핀
// progress: 0~1 (원호가 시계방향으로 채워지며 완료를 나타냄)
function drawCalibCrosshair(x, y, progress) {
  const ARM = 12;  // 십자 팔 길이
  const GAP = 4;   // 중심 공백
  const R   = 9;   // 진행 원호 반지름

  ctx.save();
  ctx.lineWidth   = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.65)";

  // 십자 팔 4개
  ctx.beginPath();
  ctx.moveTo(x - ARM, y); ctx.lineTo(x - GAP, y);
  ctx.moveTo(x + GAP, y); ctx.lineTo(x + ARM, y);
  ctx.moveTo(x, y - ARM); ctx.lineTo(x, y - GAP);
  ctx.moveTo(x, y + GAP); ctx.lineTo(x, y + ARM);
  ctx.stroke();

  // 중심 점
  ctx.beginPath();
  ctx.arc(x, y, 1.5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.65)";
  ctx.fill();

  // 배경 원 (흐림)
  ctx.beginPath();
  ctx.arc(x, y, R, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.stroke();

  // 진행 원호 (12시 방향부터 시계방향)
  if (progress > 0) {
    ctx.beginPath();
    ctx.arc(x, y, R, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.80)";
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  }

  ctx.restore();
}

// 테스트용: 소리 위치 점 + 이름 (레이블은 점 아래쪽에 표시)
function drawSoundPositions() {
  sounds.forEach(s => {
    ctx.beginPath();
    ctx.arc(s.x, s.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fill();

    ctx.font         = "11px 'Segoe UI', sans-serif";
    ctx.fillStyle    = "rgba(255,255,255,0.55)";
    ctx.textAlign    = "center";
    ctx.textBaseline = "top";
    ctx.fillText(s.name, s.x, s.y + 10);
  });
}

// ── Audio init ─────────────────────────────────────────────
function initAudio() {
  sounds = SOUND_DEFS.map(def => {
    const audio = new Audio(`sounds/${encodeURIComponent(def.file)}`);
    audio.loop = true;
    audio.volume = 0.5;
    return { audio, name: def.name, xFrac: def.xFrac, yFrac: def.yFrac, x: 0, y: 0 };
  });
  computeSoundPositions();
}

function startAudio() {
  if (audioStarted) return;
  audioStarted = true;
  permissionMsg.textContent = "";
  sounds.forEach(s => s.audio.play().catch(err => console.error(`오디오 재생 실패 [${s.name}]:`, err)));
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

  // 기준점이 화면 70% 지점이므로, 시선이 60% 위로 올라가면 소리 구간 활성화
  const threshold    = canvas.height * 0.62;
  const distAbove    = Math.max(0, threshold - gazeY);
  const focusStrength = Math.min(distAbove / threshold, 1);

  let minDist = Infinity, closestIdx = 0;
  sounds.forEach((s, i) => {
    const d = Math.sqrt((gazeX - s.x) ** 2 + (gazeY - s.y) ** 2);
    if (d < minDist) { minDist = d; closestIdx = i; }
  });

  if (focusStrength < 0.08) {
    // 중앙 또는 아래쪽: 전체 50%로 복귀
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
    const target = (i === closestIdx) ? targetFocused : 0.02;
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

    // ── 캘리브레이션 ──────────────────────────────────────
    if (!isCalibrated) {
      calibYawSum   += yaw;
      calibPitchSum += pitch;
      calibFrameCount++;

      if (calibFrameCount >= CALIB_FRAMES) {
        neutralYaw   = calibYawSum   / calibFrameCount;
        neutralPitch = calibPitchSum / calibFrameCount;
        isCalibrated = true;
        console.log(`캘리브레이션 완료 — yaw: ${neutralYaw.toFixed(4)}, pitch: ${neutralPitch.toFixed(4)}`);
      } else {
        // 캘리브레이션 중: 기준점에 십자 조준핀 표시
        drawCalibCrosshair(
          canvas.width  * 0.5,
          canvas.height * 0.70,
          calibFrameCount / CALIB_FRAMES
        );
        requestAnimationFrame(detectLoop);
        return;
      }
    }

    // 개인 정면 보정값 차감 후 증폭
    const adjYaw   = yaw   - neutralYaw;
    const adjPitch = pitch - neutralPitch;

    // 기준점: 화면 중앙이 아닌 하단 70% 지점
    const anchorX = canvas.width  * 0.50;
    const anchorY = canvas.height * 0.70;

    let pointX = anchorX + adjYaw   * canvas.width  * 1.5;
    let pointY = anchorY + adjPitch * canvas.height * 2.8;
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
